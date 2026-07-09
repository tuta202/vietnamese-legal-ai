from __future__ import annotations

import asyncio
import traceback

import reflex as rx

from legal_ui.services import get_runner, hydrate_result


EXAMPLE_QUESTION = "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?"

EXAMPLE_CHIPS = [
    {
        "label": "Thuế & đất đai",
        "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
    },
    {
        "label": "Ưu đãi đấu thầu",
        "question": "Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?",
    },
    {
        "label": "Giữ bằng cấp",
        "question": "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?",
    },
    {
        "label": "Hộ kinh doanh chuyển đổi",
        "question": "Hộ kinh doanh cần đáp ứng điều kiện gì để được hưởng hỗ trợ khi chuyển đổi thành doanh nghiệp nhỏ và vừa?",
    },
]

PROCESSING_STAGES = [
    ("Đang hiểu câu hỏi", "Hệ thống đang xác định vấn đề pháp lý chính trong câu hỏi của bạn."),
    ("Đang tìm điều luật liên quan", "Hệ thống đang tra cứu trong kho văn bản pháp luật."),
    ("Đang chọn căn cứ phù hợp", "Hệ thống đang giữ lại các điều luật có khả năng liên quan nhất."),
    ("Đang kiểm tra độ liên quan", "Hệ thống đang loại bớt các điều chỉ giống từ khóa nhưng không thật sự cần thiết."),
    ("Đang rà soát căn cứ", "Hệ thống đang bỏ các điều luật dễ gây nhầm nếu câu hỏi không cần đến chúng."),
    ("Đang chốt căn cứ pháp lý", "Hệ thống đang kiểm tra lần cuối các điều luật sẽ dùng để trả lời."),
    ("Đang kiểm tra độ đầy đủ", "Hệ thống đang xem còn khía cạnh quan trọng nào của câu hỏi chưa được bao phủ không."),
    ("Đang viết câu trả lời", "Hệ thống đang soạn câu trả lời ngắn gọn, dễ hiểu và có trích dẫn điều luật."),
    ("Đang hoàn tất", "Hệ thống đang chuẩn bị nguồn văn bản và danh sách điều luật để hiển thị."),
]

COMPLETED_STAGE_LABELS = [
    "Đã hiểu câu hỏi",
    "Đã tìm điều luật liên quan",
    "Đã chọn căn cứ phù hợp",
    "Đã kiểm tra độ liên quan",
    "Đã rà soát căn cứ",
    "Đã chốt căn cứ pháp lý",
    "Đã kiểm tra độ đầy đủ",
    "Đã viết câu trả lời",
    "Đã chuẩn bị kết quả hiển thị",
]


class QAState(rx.State):
    question: str = EXAMPLE_QUESTION
    example_chips: list[dict] = EXAMPLE_CHIPS
    example_labels: list[str] = [chip["label"] for chip in EXAMPLE_CHIPS]
    question_empty: bool = False
    loading: bool = False
    error: str = ""
    has_error: bool = False
    result: dict = {}
    has_result: bool = False
    conclusion: str = ""
    analysis: str = ""
    conclusion_blocks: list[list[dict]] = []
    analysis_blocks: list[list[dict]] = []
    has_conclusion: bool = False
    docs: list[dict] = []
    articles: list[dict] = []
    has_grounding: bool = False
    selected_article: dict = {}
    has_selected_article: bool = False
    warnings: list[str] = []
    has_warnings: bool = False
    status_title: str = ""
    status_detail: str = ""
    progress: int = 0
    completed_steps: list[str] = []
    has_completed_steps: bool = False

    @rx.event
    def update_question(self, value: str):
        self.question = value
        self.question_empty = not value.strip()

    @rx.event
    def use_example(self, question: str):
        self.question = question
        self.question_empty = not question.strip()

    @rx.event
    def use_example_label(self, label: str):
        for chip in EXAMPLE_CHIPS:
            if chip["label"] == label:
                self.question = chip["question"]
                self.question_empty = False
                return

    def _reset_progress(self):
        self.status_title = "Đang chuẩn bị"
        self.status_detail = "Hệ thống đang chuẩn bị phiên tra cứu mới."
        self.progress = 0
        self.completed_steps = []
        self.has_completed_steps = False

    def _set_stage(self, stage_index: int):
        stage_index = max(0, min(stage_index, len(PROCESSING_STAGES) - 1))
        label, detail = PROCESSING_STAGES[stage_index]
        self.status_title = label
        self.status_detail = detail
        self.progress = min(
            99,
            int(((stage_index + 1) / max(1, len(PROCESSING_STAGES))) * 99),
        )
        self.completed_steps = [COMPLETED_STAGE_LABELS[idx] for idx in range(stage_index)]
        self.has_completed_steps = bool(self.completed_steps)

    @rx.event
    async def ask(self):
        question = self.question.strip()
        if not question:
            self.error = "Vui lòng nhập câu hỏi pháp lý."
            self.question_empty = True
            return

        self.loading = True
        self.error = ""
        self.has_error = False
        self.has_result = False
        self._reset_progress()
        yield

        try:
            runner = get_runner()
            self._set_stage(0)
            yield
            route_info = await asyncio.to_thread(runner.route, question)
            route = route_info.get("route", "legal")
            if route != "legal":
                answered = await asyncio.to_thread(
                    runner.answer_without_rag,
                    "ui-1",
                    question,
                    route,
                    route_info.get("reason", ""),
                )
                self.result = hydrate_result(answered)
                self.conclusion = self.result.get("conclusion", "")
                self.analysis = self.result.get("analysis", "")
                self.conclusion_blocks = self.result.get("conclusion_blocks", [])
                self.analysis_blocks = self.result.get("analysis_blocks", [])
                self.has_conclusion = bool(self.result.get("has_conclusion", False))
                self.docs = []
                self.articles = []
                self.has_grounding = False
                self.warnings = self.result.get("warnings", [])
                self.has_warnings = bool(self.warnings)
                self.has_result = True
                self.status_title = "Đã hoàn tất"
                self.status_detail = "Câu trả lời đã sẵn sàng hiển thị."
                self.completed_steps = ["Đã hiểu câu hỏi"]
                self.has_completed_steps = True
                self.progress = 100
                return
            analysis = await asyncio.to_thread(runner.analyze, "ui-1", question)

            self._set_stage(1)
            yield
            global_row, intent_row = await asyncio.to_thread(runner.retrieve, analysis)

            self._set_stage(2)
            yield
            tiered_row, _bge_by_intent = await asyncio.to_thread(
                runner.tiered_union,
                analysis,
                global_row,
                intent_row,
            )

            legal_intents = analysis.get("legal_intents", [])
            self._set_stage(3)
            yield
            stage1_row = await asyncio.to_thread(runner.stage1, tiered_row, legal_intents)

            self._set_stage(4)
            yield
            cleaned_row = await asyncio.to_thread(runner.penalty_cleanup, stage1_row)

            self._set_stage(5)
            yield
            final_row = await asyncio.to_thread(runner.final_collective, cleaned_row, legal_intents)
            gated_row = await asyncio.to_thread(runner.enforcement_gate, final_row)

            self._set_stage(6)
            yield
            rescued_row = await asyncio.to_thread(runner.rescue, gated_row, cleaned_row, intent_row)

            self._set_stage(7)
            yield
            answered = await asyncio.to_thread(runner.generate, rescued_row)

            self._set_stage(8)
            yield
            raw = {
                **answered,
                "_debug": {
                    "legal_intents": legal_intents,
                    "sizes": {
                        "rrf60": len(global_row.get("relevant_articles", [])),
                        "tiered": len(tiered_row.get("relevant_articles", [])),
                        "stage1": len(stage1_row.get("relevant_articles", [])),
                        "penalty_cleanup": len(cleaned_row.get("relevant_articles", [])),
                        "final_collective": len(final_row.get("relevant_articles", [])),
                        "enforcement_gate": len(gated_row.get("relevant_articles", [])),
                        "rescue": len(rescued_row.get("relevant_articles", [])),
                    },
                },
            }
            self.result = hydrate_result(raw)
            self.conclusion = self.result.get("conclusion", "")
            self.analysis = self.result.get("analysis", "")
            self.conclusion_blocks = self.result.get("conclusion_blocks", [])
            self.analysis_blocks = self.result.get("analysis_blocks", [])
            self.has_conclusion = bool(self.result.get("has_conclusion", False))
            self.docs = self.result.get("docs", [])
            self.articles = self.result.get("articles", [])
            self.has_grounding = bool(self.result.get("has_grounding", False))
            self.warnings = self.result.get("warnings", [])
            self.has_warnings = bool(self.warnings)
            self.has_result = True
            self.status_title = "Đã hoàn tất"
            self.status_detail = "Câu trả lời và căn cứ pháp lý đã sẵn sàng hiển thị."
            self.completed_steps = list(COMPLETED_STAGE_LABELS)
            self.has_completed_steps = True
            self.progress = 100
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if len(self.error) < 240:
                self.error = f"{self.error}\n{traceback.format_exc(limit=1)}"
            self.has_error = True
        finally:
            self.loading = False

    @rx.event
    def set_example(self):
        self.question = EXAMPLE_QUESTION
        self.question_empty = False

    @rx.event
    def open_article(self, article: dict):
        self.selected_article = article
        self.has_selected_article = True

    @rx.event
    def close_article(self):
        self.selected_article = {}
        self.has_selected_article = False
