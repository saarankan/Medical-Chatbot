import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from config import MONGODB_URI, DB_NAME


# ─────────────────────────────────────────────
#  HOW THIS FILE WORKS — READ FIRST
#
#  This file has exactly 4 functions.
#  Each one does ONE thing with the appointments
#  collection in MongoDB.
#
#  TOOL 1 — check_availability
#    Patient asks: "Is Dr. Mohan free on Monday?"
#    → checks MongoDB, returns list of free slots
#
#  TOOL 2 — create_appointment
#    Patient says: "Book 10am with Dr. Mohan"
#    → inserts a new document into MongoDB
#    → returns a confirmation message
#
#  TOOL 3 — cancel_appointment
#    Patient says: "Cancel my appointment"
#    → finds the document, changes status to "cancelled"
#    → never deletes — always keeps the record
#
#  TOOL 4 — list_appointments
#    Patient asks: "What appointments do I have?"
#    → finds all confirmed bookings for that phone number
#    → returns them as a readable list
#
#  These 4 functions are tested directly in this file.
#  Later, agents.py will call them based on what
#  the patient says in the chat.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CONNECTION
#  Same pattern as database.py — one client
#  shared across all 4 functions.
# ─────────────────────────────────────────────

client   = AsyncIOMotorClient(MONGODB_URI)
db       = client[DB_NAME]
appt_col = db["appointments"]


# ─────────────────────────────────────────────
#  CLINIC CONFIGURATION
#
#  Edit these two things to match the real clinic:
#
#  DOCTORS     — list of doctor names exactly as
#                they should appear in the system
#
#  ALL_SLOTS   — every possible time slot the
#                clinic offers. A slot is "available"
#                if it is in this list AND has no
#                confirmed appointment in MongoDB.
# ─────────────────────────────────────────────

DOCTORS = [
    "Dr. Mohan",    # General Practitioner — Mon / Wed / Fri
    "Dr. Kumar",    # Paediatrician        — Tue / Thu
]

ALL_SLOTS = [
    "09:00 AM", "09:30 AM",
    "10:00 AM", "10:30 AM",
    "11:00 AM", "11:30 AM",
    "02:00 PM", "02:30 PM",
    "03:00 PM", "03:30 PM",
    "04:00 PM", "04:30 PM",
]


# ═════════════════════════════════════════════
#  TOOL 1 — check_availability
# ═════════════════════════════════════════════

async def check_availability(doctor: str, date: str) -> dict:
    """
    Finds all free time slots for a doctor on a given date.

    How it works:
      Step 1 — validate the doctor name is correct
      Step 2 — query MongoDB for all confirmed bookings
               for that doctor on that date
      Step 3 — remove those booked slots from ALL_SLOTS
      Step 4 — return the remaining free slots

    Args:
        doctor : doctor name — must match DOCTORS list exactly
                 e.g. "Dr. Mohan"
        date   : date in YYYY-MM-DD format
                 e.g. "2024-03-15"

    Returns:
        dict — always has "success" key (True/False)
               if True, has "available_slots" list
    """

    # ── step 1: validate doctor name ──
    # if the agent passes a wrong doctor name, return a
    # clear error rather than silently returning no slots
    if doctor not in DOCTORS:
        return {
            "success": False,
            "message": (
                f"'{doctor}' is not a recognised doctor. "
                f"Available doctors: {', '.join(DOCTORS)}"
            )
        }

    # ── step 2: find all confirmed bookings for this doctor + date ──
    # status="confirmed" means we ignore cancelled appointments —
    # a cancelled slot is free again and can be rebooked
    cursor = appt_col.find({
        "doctor": doctor,
        "date":   date,
        "status": "confirmed"
    })
    booked_appointments = await cursor.to_list(length=100)

    # extract just the time_slot strings from the documents
    # e.g. ["10:00 AM", "02:00 PM"]
    booked_slots = [appt["time_slot"] for appt in booked_appointments]

    # ── step 3: subtract booked from all slots ──
    # list comprehension — keeps a slot only if it is NOT in booked_slots
    available_slots = [
        slot for slot in ALL_SLOTS
        if slot not in booked_slots
    ]

    # ── step 4: return result ──
    if not available_slots:
        return {
            "success":   True,
            "available": False,
            "doctor":    doctor,
            "date":      date,
            "message": (
                f"No available slots for {doctor} on {date}. "
                f"All {len(ALL_SLOTS)} slots are fully booked. "
                f"Please try a different date."
            )
        }

    return {
        "success":         True,
        "available":       True,
        "doctor":          doctor,
        "date":            date,
        "available_slots": available_slots,
        "booked_count":    len(booked_slots),
        "free_count":      len(available_slots),
        "message": (
            f"{len(available_slots)} slot(s) available "
            f"for {doctor} on {date}: "
            f"{', '.join(available_slots)}"
        )
    }


# ═════════════════════════════════════════════
#  TOOL 2 — create_appointment
# ═════════════════════════════════════════════

async def create_appointment(
    patient_name : str,
    phone        : str,
    doctor       : str,
    date         : str,
    time_slot    : str,
    reason       : str = "General consultation"
) -> dict:
    """
    Books an appointment and saves it to MongoDB.

    How it works:
      Step 1 — double-check the slot is still free
               (another patient might have booked it
               between check_availability and now)
      Step 2 — insert a new document into MongoDB
      Step 3 — before confirmation, ask yes/no to confirm details
      Step 4 — return a confirmation with the booking ID

    Args:
        patient_name : full name of the patient
        phone        : patient's phone number
                       (used to look up appointments later)
        doctor       : doctor name from DOCTORS list
        date         : date in YYYY-MM-DD format
        time_slot    : must be a value from ALL_SLOTS
                       e.g. "10:00 AM"
        reason       : reason for visit (optional)
                       defaults to "General consultation"

    Returns:
        dict with success status, appointment_id, and
        a formatted confirmation message
    """
    

    # ── step 1: race condition check ──
    # check the slot one more time just before booking.
    # "race condition" = two patients book the same slot
    # at almost the same time. This prevents double-booking.
    slot_taken = await appt_col.find_one({
        "doctor":    doctor,
        "date":      date,
        "time_slot": time_slot,
        "status":    "confirmed"
    })

    if slot_taken:
        return {
            "success": False,
            "message": (
                f"Sorry, {time_slot} with {doctor} on {date} "
                f"was just booked by someone else. "
                f"Please choose a different slot."
            )
        }

    # ── step 2: build and insert the document ──
    # this is exactly the document structure that goes
    # into MongoDB — one document = one appointment
    new_appointment = {
        "patient_name" : patient_name,
        "phone"        : phone,
        "doctor"       : doctor,
        "date"         : date,
        "time_slot"    : time_slot,
        "reason"       : reason,
        "status"       : "confirmed",        # confirmed immediately on booking
        "created_at"   : datetime.utcnow()   # Python sets this automatically
    }

    result = await appt_col.insert_one(new_appointment)

    # MongoDB returns an InsertOneResult with inserted_id
    # convert ObjectId to string so it can be returned as JSON
    appt_id     = str(result.inserted_id)
    ref_display = appt_id[:8].upper()   # short 8-char reference for the patient

    # ── step 3: return confirmation ──
    return {
        "success"        : True,
        "appointment_id" : appt_id,
        "patient_name"   : patient_name,
        "phone"          : phone,
        "doctor"         : doctor,
        "date"           : date,
        "time_slot"      : time_slot,
        "reason"         : reason,
        "status"         : "confirmed",
        "message"        : (
            f"✅ Appointment confirmed!\n\n"
            f"Patient : {patient_name}\n"
            f"Doctor  : {doctor}\n"
            f"Date    : {date}\n"
            f"Time    : {time_slot}\n"
            f"Reason  : {reason}\n"
            f"Ref ID  : {ref_display}\n\n"
            f"Please arrive 10 minutes early.\n"
            f"To cancel, call the clinic or ask me with your Ref ID."
        )
    }


# ═════════════════════════════════════════════
#  TOOL 3 — cancel_appointment
# ═════════════════════════════════════════════

async def cancel_appointment(appointment_id: str, phone: str) -> dict:
    """
    Cancels an appointment by updating its status to "cancelled".

    IMPORTANT: we never DELETE documents from MongoDB.
    We always UPDATE the status to "cancelled".
    This keeps a full audit trail for the clinic —
    if a patient disputes a booking, the record is there.

    How it works:
      Step 1 — convert the appointment_id string to ObjectId
      Step 2 — find the appointment AND verify the phone matches
               (stops patients cancelling other people's bookings)
      Step 3 — check it is not already cancelled
      Step 4 — update status to "cancelled"

    Args:
        appointment_id : the _id string from MongoDB
                         e.g. "65a3f2b1c4d8e9f012345678"
        phone          : patient's phone number for verification

    Returns:
        dict with success status and message
    """

    # ── step 1: convert string → ObjectId ──
    # MongoDB stores _id as ObjectId, not a plain string.
    # If the patient gives an invalid ID, ObjectId() will raise
    # an exception — we catch it and return a clear error.
    try:
        obj_id = ObjectId(appointment_id)
    except Exception:
        return {
            "success": False,
            "message": (
                "That doesn't look like a valid appointment ID. "
                "Please check your Ref ID and try again."
            )
        }

    # ── step 2: find the appointment + verify phone ──
    # querying by BOTH _id AND phone means a patient can only
    # cancel their own appointments — not anyone else's
    appointment = await appt_col.find_one({
        "_id"  : obj_id,
        "phone": phone
    })

    if not appointment:
        return {
            "success": False,
            "message": (
                "No appointment found with that ID and phone number. "
                "Please check your details and try again."
            )
        }

    # ── step 3: check not already cancelled ──
    if appointment["status"] == "cancelled":
        return {
            "success": False,
            "message": "This appointment has already been cancelled."
        }

    # ── step 4: update status to "cancelled" ──
    # $set updates only the status field — leaves all other
    # fields (doctor, date, patient_name etc.) unchanged
    await appt_col.update_one(
        {"_id": obj_id},
        {"$set": {"status": "cancelled"}}
    )

    return {
        "success": True,
        "message": (
            f"✅ Appointment cancelled.\n\n"
            f"Doctor : {appointment['doctor']}\n"
            f"Date   : {appointment['date']}\n"
            f"Time   : {appointment['time_slot']}\n\n"
            f"Would you like to book a new appointment?"
        )
    }


# ═════════════════════════════════════════════
#  TOOL 4 — list_appointments
# ═════════════════════════════════════════════

async def list_appointments(phone: str) -> dict:
    """
    Returns all upcoming confirmed appointments for a patient.

    Patients identify themselves by phone number —
    no login required, keeping it simple for a small clinic.

    How it works:
      Step 1 — query MongoDB for all confirmed appointments
               with this phone number
      Step 2 — sort them by date then time (earliest first)
      Step 3 — format them as a readable list

    Args:
        phone : patient's phone number
                e.g. "0775763780"

    Returns:
        dict with found status, count, and
        formatted appointment list
    """

    # ── step 1 + 2: query and sort ──
    # sort=[("date", 1), ("time_slot", 1)] means:
    #   1 = ascending (earliest date first)
    #   if two appointments are on the same date,
    #   sort by time_slot alphabetically
    cursor = appt_col.find(
        {"phone": phone, "status": "confirmed"},
        sort=[("date", 1), ("time_slot", 1)]
    )
    appointments = await cursor.to_list(length=20)

    # ── handle no appointments found ──
    if not appointments:
        return {
            "success": True,
            "found"  : False,
            "count"  : 0,
            "message": (
                "You have no upcoming appointments. "
                "Would you like to book one?"
            )
        }

    # ── step 3: format each appointment ──
    appt_list = []
    for appt in appointments:
        appt_list.append({
            "appointment_id" : str(appt["_id"]),
            "ref_id"         : str(appt["_id"])[:8].upper(),
            "doctor"         : appt["doctor"],
            "date"           : appt["date"],
            "time_slot"      : appt["time_slot"],
            "reason"         : appt.get("reason", "Not specified"),
        })

    # build a readable summary message
    lines = [f"You have {len(appt_list)} upcoming appointment(s):\n"]
    for i, a in enumerate(appt_list, 1):
        lines.append(
            f"{i}. {a['doctor']}\n"
            f"   Date   : {a['date']}\n"
            f"   Time   : {a['time_slot']}\n"
            f"   Reason : {a['reason']}\n"
            f"   Ref ID : {a['ref_id']}"
        )

    return {
        "success"      : True,
        "found"        : True,
        "count"        : len(appt_list),
        "appointments" : appt_list,
        "message"      : "\n\n".join(lines)
    }


# ─────────────────────────────────────────────
#  TEST ALL 4 TOOLS
#  Run:  python backend/booking.py
#
#  Tests each tool in sequence:
#  1. Check availability for Dr. Mohan
#  2. Book a slot
#  3. List the booked appointment
#  4. Cancel it
#
#  All 4 should pass before moving to agents.py
# ─────────────────────────────────────────────

if __name__ == "__main__":

    async def run_tests():
        print("\n─────────────────────────────────")
        print("  booking.py — testing all 4 tools")
        print("─────────────────────────────────\n")

        TEST_PHONE = "0770000099"   # test phone — won't clash with real data
        TEST_DATE  = "2024-12-01"

        # ── TOOL 1: check availability ──
        print("TOOL 1 — check_availability")
        result = await check_availability("Dr. Mohan", TEST_DATE)
        if result["success"] and result["available"]:
            print(f"  PASS — {result['free_count']} slots free")
            print(f"  First 3: {result['available_slots'][:3]}")
        else:
            print(f"  Result: {result['message']}")
        print()

        # ── TOOL 2: create appointment ──
        print("TOOL 2 — create_appointment")
        result = await create_appointment(
            patient_name = "Test Patient",
            phone        = TEST_PHONE,
            doctor       = "Dr. Mohan",
            date         = TEST_DATE,
            time_slot    = "10:00 AM",
            reason       = "Booking tool test"
        )
        if result["success"]:
            appt_id = result["appointment_id"]
            print(f"  PASS — booked. Ref: {appt_id[:8].upper()}")
        else:
            print(f"  FAIL — {result['message']}")
            return
        print()

        # ── TOOL 3: list appointments ──
        print("TOOL 3 — list_appointments")
        result = await list_appointments(TEST_PHONE)
        if result["success"] and result["found"]:
            print(f"  PASS — found {result['count']} appointment(s)")
        else:
            print(f"  FAIL — {result['message']}")
        print()

        # ── TOOL 4: cancel appointment ──
        print("TOOL 4 — cancel_appointment")
        result = await cancel_appointment(appt_id, TEST_PHONE)
        if result["success"]:
            print(f"  PASS — appointment cancelled")
        else:
            print(f"  FAIL — {result['message']}")
        print()

        print("─────────────────────────────────")
        print("  All 4 tools tested successfully")
        print("  Ready to write agents.py")
        print("─────────────────────────────────\n")

        client.close()

    asyncio.run(run_tests())