from __future__ import annotations

from legal_rag.verification.enforcement_gate import drop_reason


def article(law_name: str, title: str) -> dict[str, str]:
    return {"law_name": law_name, "dieu_title": title, "content": ""}


def test_drops_specific_penalty_for_pure_compliance_when_alternative_exists() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực lao động",
        "Vi phạm quy định về thời giờ làm việc, thời giờ nghỉ ngơi",
    )
    question = "Công ty phải thông báo việc làm thêm trên 200 giờ chậm nhất khi nào?"

    assert (
        drop_reason(question, item, has_non_enforcement_alternative=True)
        == "specific_penalty_not_requested_with_alternative"
    )


def test_keeps_specific_penalty_when_question_asks_how_it_is_handled() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực lao động",
        "Vi phạm quy định về giao kết hợp đồng lao động",
    )
    question = "Nếu công ty giữ bằng cấp của nhân viên thì bị xử lý như thế nào?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_keeps_specific_penalty_for_violation_determination() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính về thuế, hóa đơn",
        "Hành vi vi phạm quy định về hóa đơn",
    )
    question = "Không xuất hóa đơn có bị coi là vi phạm pháp luật không?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_drops_generic_penalty_form_for_specific_offence_question() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực lao động",
        "Hình thức xử phạt",
    )
    question = "Không có giấy phép lao động thì công ty bị xử lý ra sao?"

    assert (
        drop_reason(question, item, has_non_enforcement_alternative=True)
        == "generic_penalty_form_not_requested"
    )


def test_keeps_generic_penalty_form_when_explicitly_requested() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính",
        "Hình thức xử phạt",
    )
    question = "Các hình thức xử phạt được áp dụng gồm những gì?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_keeps_penalty_authority_when_question_asks_official_power() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính về thuế, hóa đơn",
        "Thẩm quyền xử phạt vi phạm hành chính về thuế, hóa đơn của thanh tra",
    )
    question = "Thanh tra viên có quyền xử phạt vi phạm về thuế như thế nào?"

    assert drop_reason(question, item, has_non_enforcement_alternative=False) == ""


def test_drops_unrequested_coercion_when_alternative_exists() -> None:
    item = article(
        "Nghị định về cưỡng chế thi hành quyết định hành chính thuế",
        "Cưỡng chế bằng biện pháp kê biên tài sản",
    )
    question = "Doanh nghiệp cần chuẩn bị hồ sơ miễn thuế gồm những gì?"

    assert (
        drop_reason(question, item, has_non_enforcement_alternative=True)
        == "coercion_not_requested_with_alternative"
    )


def test_keeps_coercion_for_tax_debt_question() -> None:
    item = article(
        "Nghị định về cưỡng chế thi hành quyết định hành chính thuế",
        "Cưỡng chế bằng biện pháp kê biên tài sản",
    )
    question = "Khi công ty nợ thuế, cơ quan thuế có thể cưỡng chế bằng biện pháp nào?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_does_not_confuse_labour_discipline_with_penalty_intent() -> None:
    item = article(
        "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực lao động",
        "Vi phạm quy định về kỷ luật lao động",
    )
    question = "Công ty có được xử lý kỷ luật hành vi không có trong nội quy không?"

    assert (
        drop_reason(question, item, has_non_enforcement_alternative=True)
        == "specific_penalty_not_requested_with_alternative"
    )


def test_does_not_treat_plain_scope_article_as_penalty() -> None:
    item = article(
        "Nghị định quy định trách nhiệm nộp thuế thay cho hộ kinh doanh",
        "Đối tượng áp dụng",
    )
    question = "Nền tảng thương mại điện tử phải nộp thuế thay trong trường hợp nào?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_does_not_treat_contractual_penalty_as_administrative_penalty() -> None:
    item = article(
        "Luật Thương mại",
        "Mức phạt vi phạm",
    )
    question = "Công ty có thể yêu cầu đối tác trả tiền phạt vi phạm như thế nào?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_keeps_coercion_conditions_for_tax_debt() -> None:
    item = article(
        "Luật Quản lý thuế",
        "Trường hợp bị cưỡng chế thi hành quyết định hành chính về quản lý thuế",
    )
    question = "Người đại diện chưa nộp đủ số thuế thì bị xử lý ra sao khi xuất cảnh?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_does_not_treat_infringement_definition_as_penalty_article() -> None:
    item = {
        "law_name": "Thông tư hướng dẫn nghị định xử phạt vi phạm hành chính về sở hữu công nghiệp",
        "dieu_title": "Hành vi xâm phạm quyền đối với kiểu dáng công nghiệp",
        "content": "Khi xác định hành vi xâm phạm quyền phải tuân theo các hướng dẫn sau.",
    }
    question = "Thế nào là kiểu dáng không khác biệt đáng kể?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""


def test_does_not_treat_subject_specific_penalty_title_as_generic() -> None:
    item = {
        "law_name": "Thông tư xử phạt vi phạm hành chính trong lĩnh vực kế toán",
        "dieu_title": "Hình thức và mức xử phạt tiền đối với các hành vi vi phạm quy định về tài khoản kế toán",
        "content": "Phạt tiền đối với hành vi mở tài khoản kế toán không đúng quy định.",
    }
    question = "Vi phạm quy định về tài khoản kế toán sẽ bị xử phạt như thế nào?"

    assert drop_reason(question, item, has_non_enforcement_alternative=True) == ""
