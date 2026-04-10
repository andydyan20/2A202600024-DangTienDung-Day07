# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đặng Tiến Dũng - 2A202600024
**Nhóm:** C4
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine Similarity đo lường sự tương đồng về "hướng" giữa hai vector trong không gian đa chiều. Trong xử lý ngôn ngữ tự nhiên (NLP), nó cho biết hai đoạn văn bản có cùng ngữ cảnh hoặc ý nghĩa hay không, bất kể độ dài của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: "Làm thế nào để học lập trình Python?"
- Sentence B: "Cách tốt nhất để bắt đầu làm quen với ngôn ngữ Python là gì?"
- Tại sao tương đồng: Cả hai câu đều thể hiện ý định học lập trình Python.

**Ví dụ LOW similarity:**
- Sentence A: "Lập trình Python rất thú vị."
- Sentence B: "Hôm nay thời tiết ở Hà Nội rất đẹp."
- Tại sao khác: Câu A nói về lập trình Python, câu B nói về thời tiết.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> * Euclidean distance đo khoảng cách vật lý giữa hai điểm. Nếu một đoạn văn lặp lại cùng một nội dung nhiều lần, vector của nó sẽ rất dài và nằm xa vector của một đoạn văn ngắn có cùng nội dung, dẫn đến khoảng cách Euclidean lớn.
> * Cosine similarity chỉ đo góc giữa hai vector. Vì hai đoạn văn có cùng nội dung sẽ có vector chỉ cùng về một hướng, nên giá trị Cosine sẽ cao, giúp so sánh chính xác hơn về mặt ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *  *Trình bày phép tính:*  $\lceil \frac{document - overlap}{chunk\_size - overlap} \rceil = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil$ .
> * *Đáp án:* $\mathbf{23}$ **chunks**.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> * $num\_chunks = \lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = \mathbf{25}$ **chunks**.
- **Lý do tăng overlap:** Tăng overlap giúp duy trì ngữ cảnh giữa các chunk liên tiếp. Điều này đảm bảo các thông tin quan trọng nằm ở điểm giao cắt (như các thực thể, mối quan hệ logic) không bị mất đi khi vector hóa, giúp hệ thống retrieval (truy xuất) hiểu được thông tin liền mạch hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Incoterms 2020 (bộ quy tắc do Phòng Thương mại Quốc tế (ICC) phát hành)

**Tại sao nhóm chọn domain này?**
> Incoterms là tiêu chuẩn được sử dụng rộng rãi trong các hợp đồng mua bán hàng hóa quốc tế và nội địa để xác định rõ trách nhiệm của người mua và người bán. Việc đưa Incoterms vào embedding data giúp AI truy xuất chính xác ranh giới trách nhiệm, chi phí và rủi ro pháp lý, từ đó tối ưu hóa khả năng tư vấn và xử lý logic trong các nghiệp vụ logistics phức tạp.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | incoterms_intro.md | ICC Publication (extracted from incorterm.md) | ~1,400 | source=icc, domain=incoterms, lang=en, section=introduction |
| 2 | incoterms_any_mode_rules.md | ICC Publication (extracted from incorterm.md) | ~1,600 | source=icc, domain=incoterms, lang=en, section=any_mode |
| 3 | incoterms_sea_rules.md | ICC Publication (extracted from incorterm.md) | ~1,200 | source=icc, domain=incoterms, lang=en, section=sea_inland |
| 4 | incoterms_obligations_ab.md | ICC Publication (extracted from incorterm.md) | ~1,000 | source=icc, domain=incoterms, lang=en, section=obligations |
| 5 | incoterms_risk_cost_focus.md | ICC Publication (extracted from incorterm.md) | ~1,300 | source=icc, domain=incoterms, lang=en, section=risk_cost |

### Metadata Schema
| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | icc / team_notes | Truy vết nguồn pháp lý của thông tin |
| domain | string | incoterms | Lọc đúng domain khi chạy query nhóm |
| section | string | introduction / any_mode / sea_inland | Filter theo vùng kiến thức trong tài liệu |
| rule | string | EXW / FCA / FOB / CIF / DDP | Tăng precision cho query theo từng điều kiện giao hàng |
| lang | string | en | Đồng bộ ngôn ngữ văn bản để giảm nhiễu |
| doc_id | string | incorterm_main | Hỗ trợ quản lý/xóa tài liệu |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| incoterms_any_mode_rules.md | FixedSizeChunker (`fixed_size`) | 5 | 273.0 | Trung bình |
| incoterms_any_mode_rules.md | SentenceChunker (`by_sentences`) | 6 | 212.3 | Tốt |
| incoterms_any_mode_rules.md | RecursiveChunker (`recursive`) | 6 | 212.5 | Tốt |
| incoterms_obligations_ab.md | FixedSizeChunker (`fixed_size`) | 4 | 233.5 | Trung bình |
| incoterms_obligations_ab.md | SentenceChunker (`by_sentences`) | 2 | 435.5 | Tốt |
| incoterms_obligations_ab.md | RecursiveChunker (`recursive`) | 4 | 217.0 | Tốt |

### Strategy Của Tôi

**Loại:** HybridSemanticChunker

**Mô tả cách hoạt động:**
> Kết hợp Recursive và Semantic:
> * 1. Chia văn bản thành các đoạn lớn dựa trên cấu trúc (Paragraphs).
> * 2. Nếu đoạn văn vẫn vượt quá chunk_size, sử dụng Semantic Similarity để tìm điểm cắt tự nhiên thay vì cắt cứng theo số ký tự.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Em chọn HybridSemanticChunker vì domain Incoterms® 2020 yêu cầu sự chính xác tuyệt đối về cả cấu trúc văn bản luật lẫn ngữ nghĩa của các điều khoản đối ứng. Phương pháp này giúp duy trì tính toàn vẹn của các cặp nghĩa vụ A/B (như chuyển giao rủi ro và chi phí) bằng cách kết hợp khả năng ngắt theo phân cấp Markdown với việc phân tích mật độ ý nghĩa giữa các đoạn văn. Nhờ đó, hệ thống có thể truy xuất trọn vẹn bối cảnh của từng quy tắc giao hàng mà không bị cắt vụn thông tin kỹ thuật quan trọng như điểm giao hàng hay mức bảo hiểm tối thiểu.

**Code snippet (nếu custom):**
```python
class HybridSemanticChunker:
    """
    Kết hợp Recursive và Semantic:
    1. Chia văn bản thành các đoạn lớn dựa trên cấu trúc (Paragraphs).
    2. Nếu đoạn văn vẫn vượt quá chunk_size, sử dụng Semantic Similarity để tìm 
       điểm cắt tự nhiên thay vì cắt cứng theo số ký tự.
    """
    def __init__(
        self, 
        embedding_fn: Callable, 
        chunk_size: int = 600, 
        threshold: float = 0.82,
        separators: list[str] = ["\n\n", "\n"]
    ):
        self.embedding_fn = embedding_fn
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.separators = separators

    def chunk(self, text: str) -> list[str]:
        initial_chunks = self._recursive_split(text, self.separators)
        
        final_chunks = []
        for chunk in initial_chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                semantic_sub_chunks = self._semantic_split(chunk)
                final_chunks.extend(semantic_sub_chunks)
        
        return final_chunks

    def _semantic_split(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return [text]

        embeddings = self.embedding_fn(sentences)
        chunks = []
        buffer = [sentences[0]]
        
        for i in range(len(sentences) - 1):
            sim = compute_similarity(embeddings[i], embeddings[i+1])
            if sim < self.threshold or len(" ".join(buffer)) > self.chunk_size:
                chunks.append(" ".join(buffer))
                buffer = [sentences[i+1]]
            else:
                buffer.append(sentences[i+1])
        
        chunks.append(" ".join(buffer))
        return chunks

    def _recursive_split(self, text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = [p for p in text.split(sep) if p.strip()]
        return parts 
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| incoterms_any_mode_rules.md | best baseline (recursive) | 6 | 212.5 | 8.2/10 |
| incoterms_any_mode_rules.md | **của tôi** (recursive + semantic) | 12 | 105.1 | 9/10 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | recursive + semantic | 9.0 | Context được giữ nguyên | Số lượng chunk nhiều hơn, tốn tài nguyên hơn |
| Hiệp | Recursive + metadata filter | 9.0 | Context mạch lạc, ít nhiễu | Tốn thời gian tune separator |
| Cường | SentenceChunker | 8.5 | Câu dễ đọc, tự nhiên | Không ổn với đoạn rất dài |
| Lâm | FixedSize + overlap thấp | 8.0 | Tốc độ xử lý nhanh | Dễ mất ngữ cảnh ở ranh giới chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Strategy tốt nhất cho domain này là **Recursive + metadata filter** của Hiệp. Vì đây là các tài liệu rõ ràng về cấu trúc, có thể dễ dàng tách theo từng mục, từng điều khoản, kết hợp với metadata filter giúp tăng độ chính xác khi tìm kiếm. Nếu dùng recursive + semantic thì số lượng chunk nhiều hơn, tốn tài nguyên hơn mà độ chính xác không cao hơn nhiều.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex (?<=[.!?])\s+ để tách văn bản tại các dấu chấm, dấu hỏi hoặc chấm than mà không làm mất các ký tự này ở cuối câu. Phương pháp này xử lý tốt các edge case như văn bản không có khoảng trắng sau dấu chấm hoặc các đoạn hội thoại có nhiều dấu câu kết thúc liên tiếp.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy: chia văn bản dựa trên danh sách các dấu phân tách (separators) ưu tiên từ lớn đến nhỏ như \n\n, \n, và khoảng trắng. Kết thúc khi độ dài đoạn văn bản đã nhỏ hơn chunk_size hoặc khi danh sách separators hết, lúc đó đoạn text sẽ được trả về.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tài liệu được lưu trữ dưới dạng một danh sách các dictionary (In-memory) hoặc các bản ghi trong collection (ChromaDB) bao gồm vector, text và metadata. Độ tương đồng được tính bằng hàm compute_similarity (Cosine Similarity) thông qua tích vô hướng của hai vector đã chuẩn hóa độ dài.

**`search_with_filter` + `delete_document`** — approach:
> Áp dụng chiến lược Pre-filtering: lọc các đoạn văn bản khớp với metadata_filter trước, sau đó mới thực hiện tìm kiếm vector trên tập dữ liệu đã thu hẹp để tối ưu tốc độ. Việc xóa tài liệu được thực hiện bằng cách loại bỏ tất cả các bản ghi có doc_id tương ứng trong danh sách lưu trữ hoặc collection.

### KnowledgeBaseAgent

**`answer`** — approach:
> Prompt được thiết kế theo cấu trúc phân lớp gồm chỉ dẫn vai trò (Expert), khối dữ liệu bối cảnh (Context) và các ràng buộc phản hồi (Constraints) để đảm bảo tính trung thực. Context được inject bằng cách nối các chunk văn bản có độ tương đồng cao nhất thành một khối văn bản duy nhất thông qua f-string, giúp "ép" mô hình chỉ đưa ra câu trả lời dựa trên những bằng chứng xác thực từ tài liệu Incoterms® 2020.

### Test Results

```
# Paste output of: pytest tests/ -v

================================== test session starts ===================================
platform darwin -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- /Users/dungdang/Documents/code/vinuni/day7/2A202600024-DangTienDung-Day07/venv/bin/python3.14
cachedir: .pytest_cache
rootdir: /Users/dungdang/Documents/code/vinuni/day7/2A202600024-DangTienDung-Day07
plugins: anyio-4.13.0
collected 42 items                                                                       

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED       [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED      [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED             [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED        [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED    [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED              [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED             [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED       [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED        [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED       [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED  [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED   [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

=================================== 42 passed in 0.03s ===================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Seller delivers when goods are on board. | Risk transfers at shipment point for FOB. | High | 0.85 | Đúng |
| 2 | EXW has minimum obligations. | DDP gives seller maximum obligations. | High | 0.72 | Đúng |
| 3 | Incoterms do not regulate payment. | The weather is hot today. | Low | 0.15 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp câu số 2 có similarity khá cao dù EXW và DDP là hai thái cực đối lập. Điều này cho thấy embeddings nhận diện được chúng cùng nằm trong cấu trúc so sánh về obligations trong cùng một domain.
---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What do Incoterms 2020 rules mainly regulate between seller and buyer? | Incoterms mainly regulate obligations, risk transfer point, and allocation of costs between seller and buyer. |
| 2 | Which Incoterms rule is the only one requiring the seller to unload goods at destination? | DPU is the only Incoterms 2020 rule that requires the seller to unload at destination. |
| 3 | What do Incoterms rules explicitly NOT do regarding ownership? | Incoterms do not regulate transfer of property/title/ownership of the goods. |
| 4 | In FOB, when does risk transfer from seller to buyer? | In FOB, risk transfers when the goods are delivered on board the vessel at the named port of shipment. |
| 5 | (Filter by rule=CPT) Under CPT, does the seller guarantee goods arrive in sound condition at destination? | No. Under CPT, risk transfers when goods are handed to the carrier; seller does not guarantee arrival condition at destination. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What do Incoterms regulate? | `incoterms_intro.md` | 0.92 | Yes | Nghĩa vụ, rủi ro và chi phí. |
| 2 | Rule requiring unloading? | `incoterms_any_mode_rules.md` | 0.96 | Yes | Quy tắc DPU. |
| 3 | Not regulate ownership? | `incoterms_intro.md` | 0.91 | Yes | Incoterms không quy định về chuyển giao quyền sở hữu. |
| 4 | FOB risk transfer? | `incoterms_sea_rules.md` | 0.94 | Yes | Khi hàng đã đặt lên tàu (on board). |
| 5 | (Filter) CPT seller guarantee? | `incoterms_any_mode_rules.md` | 0.89 | Yes | Không, rủi ro chuyển giao khi giao cho carrier. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Việc chia nhỏ dữ liệu thành các tệp chuyên biệt theo logic (Sea vs Any Mode) hoạt động như một lớp lọc dữ liệu thô cực kỳ hiệu quả trước khi đưa vào hệ thống RAG.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Cách nhóm khác xử lý các câu hỏi phủ định (negative questions) bằng cách tìm kiếm các đoạn văn bản chứa từ khóa phủ định (ví dụ: "NOT", "does not") để loại trừ thông tin sai lệch.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Em sẽ tinh chỉnh `RecursiveChunker` để nhận diện các ký hiệu liệt kê (`-`, `*`) tốt hơn, tránh việc cắt ngang các danh sách nghĩa vụ A1-A10 trong file `incoterms_obligations_ab.md`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 25 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **90 / 100** |
