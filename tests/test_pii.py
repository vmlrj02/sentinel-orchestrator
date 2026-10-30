from sentinel.pii import mask_pii, mask_text


def test_masks_contact_details_and_ids():
    text = "Reach me at priya.r@example.com or +91 98765 43210, account 0012345678901"
    masked = mask_text(text)
    assert "example.com" not in masked and "98765" not in masked and "0012345678901" not in masked


def test_masks_name_field_in_structures_and_summaries():
    data = {"person_name": "Priya Shah", "person_income": 55000,
            "messages": ['{"person_name": "Priya Shah", "loan_amnt": 12000}', "- Name: Priya Shah\n- Age: 29"]}
    masked = mask_pii(data=data)
    assert masked["person_name"] == "[NAME]" and masked["person_income"] == 55000
    assert "Priya" not in str(masked)


def test_leaves_loan_numbers_alone():
    assert mask_text("loan of 12000 at 11.2% for 36 months") == "loan of 12000 at 11.2% for 36 months"
