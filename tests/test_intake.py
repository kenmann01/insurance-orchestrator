from underwriting.agents.intake import intake_agent
from underwriting.llm import ScriptedClient
from underwriting.models import RawApplication


def _client(reply: str) -> ScriptedClient:
    return ScriptedClient({"intake agent": reply})


def _raw(**overrides) -> RawApplication:
    data = {
        "applicant_name": "Dana Whitfield",
        "date_of_birth": "1990-04-12",
        "product": "auto",
        "coverage_amount": 25000.0,
        "stated_income": 80000.0,
    }
    data.update(overrides)
    return RawApplication(**data)


def test_intake_happy_path_keeps_normalized_values():
    reply = (
        '{"applicant_name": "Dana Whitfield", "date_of_birth": "1990-04-12", '
        '"product": "auto", "coverage_amount": 25000.0, "stated_income": 80000.0, '
        '"missing_fields": [], "notes": []}'
    )
    result = intake_agent(_raw(), llm=_client(reply))
    assert result.missing_fields == []
    assert result.applicant_name == "Dana Whitfield"
    assert result.date_of_birth == "1990-04-12"
    assert result.product == "auto"
    assert result.coverage_amount == 25000.0
    assert result.stated_income == 80000.0


def test_intake_blank_date_of_birth_is_listed():
    reply = (
        '{"applicant_name": "Dana Whitfield", "date_of_birth": null, '
        '"product": "auto", "coverage_amount": 25000.0, "stated_income": 80000.0, '
        '"missing_fields": ["date_of_birth"], "notes": []}'
    )
    result = intake_agent(_raw(date_of_birth=None), llm=_client(reply))
    assert result.missing_fields == ["date_of_birth"]
    assert result.date_of_birth is None


def test_intake_malformed_date_is_nulled_and_flagged():
    reply = (
        '{"applicant_name": "Dana Whitfield", "date_of_birth": "April 12, 1990", '
        '"product": "auto", "coverage_amount": 25000.0, "stated_income": 80000.0, '
        '"missing_fields": [], "notes": []}'
    )
    result = intake_agent(_raw(), llm=_client(reply))
    assert result.date_of_birth is None
    assert "date_of_birth" in result.missing_fields
    assert any("ISO" in note for note in result.notes)


def test_intake_code_catches_missing_field_the_model_omitted():
    reply = (
        '{"applicant_name": "Dana Whitfield", "date_of_birth": null, '
        '"product": "auto", "coverage_amount": 25000.0, "stated_income": 80000.0, '
        '"missing_fields": [], "notes": ["looks fine"]}'
    )
    result = intake_agent(_raw(date_of_birth=None), llm=_client(reply))
    assert "date_of_birth" in result.missing_fields
    assert "looks fine" in result.notes
    assert any("corrected" in note for note in result.notes)
