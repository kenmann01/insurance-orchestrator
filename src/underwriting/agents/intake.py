"""Intake agent: normalize the raw application and flag missing fields.

The model proposes a normalized record, but code has the final word: it
recomputes missing_fields from the parsed result and nulls any value that
fails a structural check. The model is treated as untrusted output.
"""

from datetime import datetime

from underwriting.llm import LLMClient, parse_json_model
from underwriting.models import IntakeResult, RawApplication

INTAKE_SYSTEM = (
    "You are the intake agent. You normalize raw insurance applications into a "
    "structured intake record.\n"
    "Return ONLY a JSON object with exactly these fields:\n"
    '- "applicant_name": string\n'
    '- "date_of_birth": string in ISO YYYY-MM-DD format, or null if blank\n'
    '- "product": one of "auto", "home", "life", or null if blank\n'
    '- "coverage_amount": number, or null if blank\n'
    '- "stated_income": number, or null if blank\n'
    '- "missing_fields": list of critical fields the applicant left blank; the '
    "critical fields are date_of_birth, product, coverage_amount\n"
    '- "notes": list of strings\n'
    "Do not invent values for blank fields: use null and list the field in "
    "missing_fields."
)


def intake_agent(raw: RawApplication, *, llm: LLMClient) -> IntakeResult:
    user = "Application JSON:\n" + raw.model_dump_json()
    response = llm.complete(INTAKE_SYSTEM, user)
    result = parse_json_model(llm, INTAKE_SYSTEM, user, response.text, IntakeResult)

    model_missing = list(result.missing_fields)
    notes = list(result.notes)
    missing: list[str] = []

    dob = result.date_of_birth
    if dob is None or dob.strip() == "":
        missing.append("date_of_birth")
    else:
        try:
            datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            result.date_of_birth = None
            missing.append("date_of_birth")
            notes.append("date_of_birth was not an ISO YYYY-MM-DD date")

    if result.product not in ("auto", "home", "life"):
        result.product = None
        missing.append("product")

    if result.coverage_amount is None or result.coverage_amount <= 0:
        result.coverage_amount = None
        missing.append("coverage_amount")

    caught = sorted(set(missing) - set(model_missing))
    if caught:
        notes.append(
            "intake code corrected missing_fields: added " + ", ".join(caught)
        )

    result.missing_fields = missing
    result.notes = notes
    return result
