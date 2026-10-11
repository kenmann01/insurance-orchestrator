from underwriting.models import CaseRecord, RawApplication


def _raw() -> RawApplication:
    return RawApplication(
        applicant_name="Dana Whitfield",
        product="auto",
        coverage_amount=25000.0,
    )


def test_case_record_new_unique_ids_and_default_status():
    a = CaseRecord.new(_raw())
    b = CaseRecord.new(_raw())
    assert a.case_id != b.case_id
    assert a.status == "in_progress"
    assert a.intake is None
    assert a.enrichment is None
    assert a.risk is None
    assert a.recommendation is None


def test_case_record_keeps_raw_application():
    raw = _raw()
    record = CaseRecord.new(raw)
    assert record.raw.applicant_name == "Dana Whitfield"
    assert record.raw.coverage_amount == 25000.0


def test_raw_application_optional_fields_default_to_none():
    raw = _raw()
    assert raw.date_of_birth is None
    assert raw.address is None
    assert raw.stated_income is None
    assert raw.claims_history_text is None
