from __future__ import annotations

import reflex as rx

from legal_ui.state import QAState


NAVY = "#111827"
TEAL = "#2563eb"
TEAL_LIGHT = "#8b5cf6"
GREEN_SOFT = "rgba(239, 246, 255, 0.78)"
SLATE = "#334155"
MUTED = "#64748b"
BORDER = "rgba(99, 102, 241, 0.18)"
PANEL = "rgba(255, 255, 255, 0.68)"
PAGE_BG = (
    "radial-gradient(circle at 12% 8%, rgba(96, 165, 250, 0.30), transparent 30%), "
    "radial-gradient(circle at 82% 2%, rgba(168, 85, 247, 0.22), transparent 28%), "
    "linear-gradient(135deg, #f8fbff 0%, #eef4ff 44%, #f7f3ff 100%)"
)


def page_card(*children, **props) -> rx.Component:
    return rx.vstack(
        *children,
        spacing=props.pop("spacing", "4"),
        align=props.pop("align", "start"),
        width="100%",
        background=props.pop("background", PANEL),
        border=props.pop("border", f"1px solid {BORDER}"),
        border_radius=props.pop("border_radius", "8px"),
        box_shadow=props.pop(
            "box_shadow",
            "18px 18px 45px rgba(30, 41, 59, 0.08), -14px -14px 36px rgba(255, 255, 255, 0.82)",
        ),
        padding=props.pop("padding", "22px"),
        backdrop_filter=props.pop("backdrop_filter", "blur(18px)"),
        **props,
    )


def section_title(title: str, subtitle: str = "") -> rx.Component:
    return rx.vstack(
        rx.heading(title, size="4", color=NAVY),
        rx.cond(subtitle != "", rx.text(subtitle, size="2", color=MUTED, line_height="1.6"), rx.fragment()),
        spacing="1",
        align="start",
        width="100%",
    )


def feature_chip(icon: str, text: str) -> rx.Component:
    return rx.hstack(
        rx.icon(icon, size=15, color=TEAL),
        rx.text(text, size="2", color=SLATE, weight="medium"),
        spacing="2",
        align="center",
        padding="8px 10px",
        border=f"1px solid {BORDER}",
        border_radius="8px",
        background="rgba(255, 255, 255, 0.56)",
        box_shadow="inset 0 1px 0 rgba(255, 255, 255, 0.65)",
    )


def hero_header() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon("scale", size=26, color="white"),
            width="48px",
            height="48px",
            display="flex",
            align_items="center",
            justify_content="center",
            border_radius="8px",
            background=f"linear-gradient(135deg, {TEAL}, {TEAL_LIGHT})",
            box_shadow="0 16px 34px rgba(79, 70, 229, 0.24)",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.heading(
                "Trợ lý pháp lý Việt Nam",
                font_size=["24px", "30px"],
                font_weight="800",
                letter_spacing="0",
                color=NAVY,
                line_height="1.12",
            ),
            rx.text(
                "Hỏi đáp pháp lý có trích dẫn điều luật và nguồn văn bản.",
                font_size=["14px", "15px"],
                color=MUTED,
                line_height="1.45",
            ),
            spacing="1",
            align="start",
            min_width="0",
        ),
        align="center",
        spacing="3",
        width="100%",
        padding="12px 0 4px 0",
    )


def question_panel() -> rx.Component:
    return page_card(
        section_title(
            "Câu hỏi pháp lý",
            "Nhập câu hỏi tiếng Việt. Hệ thống sẽ tìm văn bản liên quan trước khi trả lời.",
        ),
        rx.text_area(
            value=QAState.question,
            on_change=QAState.update_question,
            min_height=["116px", "152px"],
            resize="vertical",
            placeholder="Ví dụ: Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
            width="100%",
            border_color="rgba(99, 102, 241, 0.28)",
            border_radius="8px",
            font_size="15px",
            line_height="1.6",
            background="rgba(255, 255, 255, 0.72)",
            _focus={
                "border_color": TEAL,
                "box_shadow": "0 0 0 4px rgba(99, 102, 241, 0.16)",
            },
        ),
        rx.vstack(
            rx.hstack(
                rx.icon("sparkles", size=15, color=TEAL),
                rx.text("Câu hỏi mẫu", size="2", color=SLATE, weight="medium"),
                spacing="2",
                align="center",
            ),
            rx.select(
                QAState.example_labels,
                placeholder="Chọn một câu hỏi mẫu để thử",
                on_change=QAState.use_example_label,
                width="100%",
                size="3",
                variant="surface",
                color_scheme="indigo",
                background="rgba(255, 255, 255, 0.74)",
                border=f"1px solid {BORDER}",
                box_shadow="inset 0 1px 0 rgba(255, 255, 255, 0.72)",
                _focus={
                    "border_color": TEAL,
                    "box_shadow": "0 0 0 4px rgba(99, 102, 241, 0.16)",
                },
            ),
            rx.text(
                "Chọn mẫu sẽ tự điền câu hỏi vào ô bên trên.",
                size="1",
                color=MUTED,
            ),
            spacing="2",
            align="start",
            width="100%",
            padding="10px",
            border=f"1px solid {BORDER}",
            border_radius="8px",
            background="rgba(255, 255, 255, 0.54)",
            box_shadow="inset 0 1px 0 rgba(255, 255, 255, 0.65)",
        ),
        rx.button(
            rx.icon("search", size=18),
            rx.cond(QAState.loading, rx.text("Đang phân tích..."), rx.text("Phân tích và trả lời")),
            on_click=QAState.ask,
            loading=QAState.loading,
            disabled=QAState.question_empty,
            width="100%",
            height="44px",
            border_radius="8px",
            background=f"linear-gradient(135deg, {TEAL}, {TEAL_LIGHT})",
            box_shadow="0 16px 34px rgba(79, 70, 229, 0.28)",
            color="white",
        ),
        spacing="4",
        position="sticky",
        top="12px",
    )


def skeleton_line(width: str = "100%", height: str = "12px") -> rx.Component:
    return rx.box(
        height=height,
        width=width,
        border_radius="8px",
        background="linear-gradient(90deg, #dbeafe 0%, #f5f3ff 48%, #dbeafe 100%)",
    )


def empty_result_state() -> rx.Component:
    return page_card(
        rx.center(
            rx.box(
                rx.icon("scale", size=34, color="white"),
                width="64px",
                height="64px",
                display="flex",
                align_items="center",
                justify_content="center",
                border_radius="999px",
                background=f"linear-gradient(135deg, {TEAL}, {TEAL_LIGHT})",
                box_shadow="0 14px 30px rgba(79, 70, 229, 0.24)",
            ),
            width="100%",
        ),
        rx.vstack(
            rx.heading("Kết quả phân tích sẽ hiển thị tại đây", size="5", color=NAVY, text_align="center"),
            rx.text(
                "Sau khi tra cứu, hệ thống sẽ hiển thị kết luận ngắn gọn, lập luận pháp lý, văn bản nguồn và danh sách điều luật được viện dẫn.",
                color=MUTED,
                text_align="center",
                line_height="1.7",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        background="rgba(255, 255, 255, 0.58)",
        border="1px dashed rgba(99, 102, 241, 0.28)",
        min_height="360px",
        justify="center",
        align="center",
        spacing="5",
    )


def completed_step(label: str) -> rx.Component:
    return rx.hstack(
        rx.icon("check", size=14, color=TEAL),
        rx.text(label, size="2", color=MUTED),
        spacing="2",
        align="center",
        width="100%",
    )


def loading_result_state() -> rx.Component:
    return page_card(
        rx.hstack(
            rx.spinner(size="3", color=TEAL),
            rx.vstack(
                rx.heading(QAState.status_title, size="5", color=NAVY),
                rx.text(QAState.status_detail, color=MUTED, line_height="1.6"),
                spacing="1",
                align="start",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        rx.box(
            rx.box(
                height="8px",
                width=QAState.progress.to_string() + "%",
                background=f"linear-gradient(135deg, {TEAL}, {TEAL_LIGHT})",
                border_radius="999px",
            ),
            height="8px",
            width="100%",
            background="rgba(199, 210, 254, 0.55)",
            border_radius="999px",
            overflow="hidden",
        ),
        rx.hstack(
            rx.text("Tiến độ", size="2", color=MUTED),
            rx.spacer(),
            rx.text(QAState.progress.to_string(), "%", size="2", weight="medium", color=NAVY),
            width="100%",
        ),
        rx.cond(
            QAState.has_completed_steps,
            rx.vstack(
                rx.text("Đã xong", size="2", weight="medium", color=NAVY),
                rx.foreach(QAState.completed_steps, completed_step),
                spacing="2",
                align="start",
                width="100%",
            ),
            rx.text(
                "Quá trình có thể mất một lúc vì hệ thống đang đối chiếu nhiều nguồn luật.",
                size="2",
                color=MUTED,
                line_height="1.6",
            ),
        ),
        rx.vstack(
            skeleton_line("42%"),
            skeleton_line("86%"),
            skeleton_line("72%"),
            skeleton_line("54%"),
            spacing="3",
            width="100%",
            margin_top="8px",
        ),
        background="rgba(255, 255, 255, 0.58)",
        border="1px dashed rgba(99, 102, 241, 0.28)",
        min_height=["420px", "calc(100vh - 220px)"],
    )


def legal_source_card(article: dict) -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.badge(article["article_number"], color_scheme="indigo", variant="soft"),
                    rx.text(
                        article["article_title"],
                        weight="bold",
                        color=NAVY,
                        white_space="normal",
                        overflow="hidden",
                        text_overflow="ellipsis",
                        flex="1",
                        min_width="0",
                    ),
                    spacing="2",
                    align="center",
                    width="100%",
                ),
                rx.text(
                    article["law_type"],
                    " ",
                    article["law_name"],
                    size="2",
                    color=MUTED,
                    line_height="1.5",
                    white_space="normal",
                ),
                rx.text(
                    article["content_preview"],
                    size="2",
                    color=SLATE,
                    line_height="1.7",
                    white_space="pre-wrap",
                ),
                rx.hstack(
                    rx.text(
                        article["article_ref"],
                        size="1",
                        color=MUTED,
                        white_space="normal",
                        line_height="1.4",
                        flex="1",
                        min_width="0",
                    ),
                    rx.text("Bấm để xem đầy đủ", size="1", color=TEAL, weight="medium", white_space="nowrap"),
                    spacing="3",
                    align="center",
                    width="100%",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            on_click=QAState.open_article(article),
            cursor="pointer",
            width="100%",
        ),
        spacing="2",
        align="start",
        padding="14px",
        border=f"1px solid {BORDER}",
        border_radius="8px",
        background="rgba(255, 255, 255, 0.72)",
        box_shadow="8px 8px 20px rgba(30, 41, 59, 0.06), -8px -8px 20px rgba(255, 255, 255, 0.78)",
        width="100%",
        id=article["anchor_id"],
        scroll_margin_top="16px",
    )


def article_dialog() -> rx.Component:
    return rx.cond(
        QAState.has_selected_article,
        rx.box(
            rx.box(
                rx.vstack(
                    rx.box(
                        rx.hstack(
                            rx.vstack(
                                rx.badge(
                                    QAState.selected_article.get("article_number", ""),
                                    color_scheme="indigo",
                                    variant="soft",
                                ),
                                rx.heading(
                                    QAState.selected_article.get("article_title", ""),
                                    size="4",
                                    color=NAVY,
                                ),
                                rx.text(
                                    QAState.selected_article.get("law_type", ""),
                                    " ",
                                    QAState.selected_article.get("law_name", ""),
                                    size="2",
                                    color=MUTED,
                                    line_height="1.5",
                                ),
                                spacing="2",
                                align="start",
                                width="100%",
                            ),
                            rx.button(
                                rx.icon("x", size=16),
                                on_click=QAState.close_article,
                                variant="soft",
                                color_scheme="indigo",
                                size="2",
                                border=f"1px solid {BORDER}",
                                background="rgba(238, 242, 255, 0.72)",
                            ),
                            justify="between",
                            align="start",
                            width="100%",
                        ),
                        padding="22px 22px 14px 22px",
                        width="100%",
                    ),
                    rx.box(
                        rx.text(
                            QAState.selected_article.get("content", ""),
                            size="2",
                            color=SLATE,
                            line_height="1.75",
                            white_space="pre-wrap",
                        ),
                        width="100%",
                        flex="1",
                        min_height="0",
                        overflow_y="auto",
                        padding="0 22px 14px 22px",
                    ),
                    rx.box(
                        rx.text(
                            QAState.selected_article.get("article_ref", ""),
                            size="1",
                            color=MUTED,
                            white_space="normal",
                            line_height="1.5",
                            width="100%",
                        ),
                        width="100%",
                        flex_shrink="0",
                        overflow="visible",
                        padding="12px 22px 20px 22px",
                        border_top=f"1px solid {BORDER}",
                        background="rgba(248, 250, 255, 0.92)",
                    ),
                    spacing="0",
                    align="start",
                    width="100%",
                    height="100%",
                ),
                width=["94vw", "900px"],
                height=["86vh", "84vh"],
                overflow="hidden",
                padding="0",
                border_radius="12px",
                background="rgba(255, 255, 255, 0.92)",
                box_shadow="0 28px 80px rgba(55, 48, 163, 0.24)",
                backdrop_filter="blur(18px)",
                display="flex",
                flex_direction="column",
            ),
            position="fixed",
            inset="0",
            z_index="40",
            background="rgba(15, 23, 42, 0.44)",
            display="flex",
            align_items="center",
            justify_content="center",
            padding="16px",
        ),
        rx.fragment(),
    )


def source_doc_card(doc: dict) -> rx.Component:
    return rx.hstack(
        rx.icon("book-open", size=18, color=TEAL),
        rx.vstack(
            rx.text(doc["law_type"], size="2", color=MUTED),
            rx.text(doc["law_name"], weight="medium", color=NAVY),
            rx.code(doc["law_id"], color_scheme="indigo"),
            spacing="1",
            align="start",
        ),
        spacing="3",
        align="start",
        padding="12px",
        border=f"1px solid {BORDER}",
        border_radius="8px",
        background="rgba(255, 255, 255, 0.72)",
        box_shadow="8px 8px 20px rgba(30, 41, 59, 0.06), -8px -8px 20px rgba(255, 255, 255, 0.78)",
        width="100%",
    )


def disclaimer_card() -> rx.Component:
    return rx.callout(
        "Thông tin chỉ mang tính hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức.",
        icon="triangle-alert",
        color_scheme="amber",
        width="100%",
    )


def answer_segment(segment: dict) -> rx.Component:
    return rx.cond(
        segment["kind"] == "citation",
        rx.el.span(
            segment["text"],
            color=TEAL,
            font_weight="700",
            text_decoration="underline",
            text_decoration_style="dotted",
            text_underline_offset="3px",
            cursor="pointer",
            on_click=QAState.open_article(segment["article"]),
        ),
        rx.cond(
            segment["kind"] == "heading",
            rx.el.div(
                segment["text"],
                color=NAVY,
                font_weight="800",
                font_size="18px",
                line_height="1.35",
                margin_top="10px",
                margin_bottom="6px",
            ),
            rx.cond(
                segment["kind"] == "bold",
                rx.el.span(segment["text"], color=NAVY, font_weight="700"),
                rx.cond(
                    segment["kind"] == "code",
                    rx.el.code(
                        segment["text"],
                        color=NAVY,
                        background="rgba(99, 102, 241, 0.10)",
                        border_radius="4px",
                        padding="1px 4px",
                    ),
                    rx.cond(
                        segment["kind"] == "bullet",
                        rx.el.span(segment["text"], color=TEAL, font_weight="700", padding_right="6px"),
                        rx.el.span(segment["text"], color=SLATE),
                    ),
                ),
            ),
        ),
    )


def answer_line(segments: list[dict]) -> rx.Component:
    return rx.box(
        rx.foreach(segments, answer_segment),
        line_height="1.8",
        width="100%",
        white_space="normal",
        overflow_wrap="anywhere",
    )


def grounded_answer(blocks: list[list[dict]]) -> rx.Component:
    return rx.vstack(
        rx.foreach(blocks, answer_line),
        spacing="2",
        align="start",
        width="100%",
    )


def answer_result_state() -> rx.Component:
    return rx.vstack(
        rx.cond(
            QAState.has_conclusion,
            rx.vstack(
                page_card(
                    section_title("Kết luận"),
                    grounded_answer(QAState.conclusion_blocks),
                    background=GREEN_SOFT,
                    border="1px solid rgba(99, 102, 241, 0.18)",
                    box_shadow="none",
                ),
                page_card(
                    section_title("Phân tích pháp lý"),
                    grounded_answer(QAState.analysis_blocks),
                    box_shadow="none",
                    min_height=["260px", "320px"],
                ),
                spacing="4",
                align="start",
                width="100%",
            ),
            page_card(
                section_title("Câu trả lời"),
                grounded_answer(QAState.analysis_blocks),
                box_shadow="none",
                min_height=["360px", "calc(100vh - 280px)"],
            ),
        ),
        rx.cond(
            QAState.has_grounding,
            rx.vstack(
                rx.vstack(
                    section_title("Nguồn văn bản", "Các văn bản mà câu trả lời đang dựa vào."),
                    rx.foreach(QAState.docs, source_doc_card),
                    spacing="3",
                    align="start",
                    width="100%",
                ),
                rx.vstack(
                    section_title("Căn cứ pháp lý", "Danh sách điều luật liên quan được hệ thống giữ lại."),
                    rx.foreach(QAState.articles, legal_source_card),
                    spacing="3",
                    align="start",
                    width="100%",
                ),
                spacing="4",
                align="start",
                width="100%",
            ),
            rx.fragment(),
        ),
        rx.cond(
            QAState.has_warnings,
            page_card(
                section_title("Lưu ý"),
                rx.vstack(
                    rx.foreach(QAState.warnings, lambda text: rx.hstack(rx.icon("triangle-alert", size=15, color="#a16207"), rx.text(text, size="2", color="#713f12"))),
                    spacing="2",
                    align="start",
                ),
                background="#fffdf5",
                border="1px solid rgba(161, 98, 7, 0.18)",
                box_shadow="none",
            ),
            rx.fragment(),
        ),
        rx.cond(QAState.has_grounding, disclaimer_card(), rx.fragment()),
        spacing="4",
        align="start",
        width="100%",
        min_height=["420px", "calc(100vh - 220px)"],
    )


def error_result_state() -> rx.Component:
    return page_card(
        rx.callout(
            "Không thể tra cứu lúc này. Vui lòng thử lại hoặc kiểm tra backend RAG.",
            icon="triangle-alert",
            color_scheme="red",
            width="100%",
        ),
        rx.text(QAState.error, size="2", color="#991b1b", white_space="pre-wrap", line_height="1.6"),
        background="#fff7f7",
        border="1px solid rgba(220, 38, 38, 0.18)",
        min_height="360px",
    )


def result_panel() -> rx.Component:
    return rx.cond(
        QAState.loading,
        loading_result_state(),
        rx.cond(
            QAState.has_error,
            error_result_state(),
            rx.cond(QAState.has_result, answer_result_state(), empty_result_state()),
        ),
    )


def footer_disclaimer() -> rx.Component:
    return rx.text(
        "Hệ thống hỗ trợ tra cứu và tổng hợp căn cứ pháp lý. Khi áp dụng vào vụ việc cụ thể, cần kiểm tra văn bản gốc và bối cảnh thực tế.",
        size="2",
        color=MUTED,
        text_align="center",
        line_height="1.6",
        width="100%",
    )


def page_shell(*children) -> rx.Component:
    return rx.box(
        rx.vstack(
            *children,
            spacing="4",
            width="100%",
            max_width="1360px",
            margin="0 auto",
            padding=["14px", "20px"],
            position="relative",
            z_index="1",
        ),
        article_dialog(),
        min_height="100vh",
        background=PAGE_BG,
    )


def index() -> rx.Component:
    return page_shell(
        hero_header(),
        rx.grid(
            question_panel(),
            result_panel(),
            grid_template_columns=["1fr", "minmax(340px, 400px) minmax(0, 1fr)"],
            gap=["16px", "22px"],
            align_items="start",
            width="100%",
        ),
        footer_disclaimer(),
    )


app = rx.App()
app.add_page(index, route="/", title="Trợ lý pháp lý Việt Nam")

