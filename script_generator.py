import random
from pathlib import Path
from typing import Dict, List, Tuple

# ==========================================
# cấu hình
# ==========================================

SEED = 42
random.seed(SEED)

# tổng số câu cần sinh
N_SENTENCES = 100

# độ dài mục tiêu theo số từ
MIN_WORDS = 28
MAX_WORDS = 52

# file output
OUTPUT_FILE = "asr_script.txt"

# phân bổ domain, tổng phải bằng 100
DOMAIN_COUNTS = {
    "doi_song": 17,
    "hoc_tap": 15,
    "cong_nghe_ai": 15,
    "lich_trinh": 13,
    "chi_dan_yeu_cau": 12,
    "mua_sam_thanh_toan": 10,
    "goi_dien_dat_lich": 9,
    "ke_chuyen_mo_ta": 9,
}

assert sum(DOMAIN_COUNTS.values()) == N_SENTENCES, "tổng domain_counts phải bằng n_sentences"

# ==========================================
# đổi số thành chữ
# ==========================================

DIGITS = {
    0: "không",
    1: "một",
    2: "hai",
    3: "ba",
    4: "bốn",
    5: "năm",
    6: "sáu",
    7: "bảy",
    8: "tám",
    9: "chín",
}

def read_digit_sequence(number_str: str) -> str:
    return " ".join(DIGITS[int(ch)] for ch in number_str if ch.isdigit())

def read_two_digits(n: int) -> str:
    if n < 10:
        return DIGITS[n]

    tens = n // 10
    ones = n % 10

    if tens == 1:
        if ones == 0:
            return "mười"
        if ones == 5:
            return "mười lăm"
        return f"mười {DIGITS[ones]}"

    tens_word = f"{DIGITS[tens]} mươi"
    if ones == 0:
        return tens_word
    if ones == 1:
        return f"{tens_word} mốt"
    if ones == 4:
        return f"{tens_word} tư"
    if ones == 5:
        return f"{tens_word} lăm"
    return f"{tens_word} {DIGITS[ones]}"

def read_three_digits(n: int, full: bool = False) -> str:
    hundreds = n // 100
    rest = n % 100
    parts: List[str] = []

    if hundreds > 0:
        parts.append(f"{DIGITS[hundreds]} trăm")
    elif full and rest > 0:
        parts.append("không trăm")

    if rest == 0:
        return " ".join(parts).strip()

    if rest < 10:
        if hundreds > 0 or full:
            parts.append("linh")
        parts.append(DIGITS[rest])
        return " ".join(parts).strip()

    parts.append(read_two_digits(rest))
    return " ".join(parts).strip()

def number_to_vietnamese(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_vietnamese(-n)

    scales = ["", "nghìn", "triệu", "tỷ"]
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000

    parts = []
    for i in range(len(groups) - 1, -1, -1):
        group = groups[i]
        if group == 0:
            continue

        full = i < len(groups) - 1 and groups[i + 1] > 0 and group < 100
        group_text = read_three_digits(group, full=full).strip()
        scale = scales[i]

        if group_text:
            parts.append(f"{group_text} {scale}".strip())

    return " ".join(parts).strip()

# ==========================================
# dữ liệu động
# ==========================================

def money_text() -> str:
    amounts = [
        12000, 18000, 25000, 35000, 50000, 68000, 75000, 99000,
        120000, 150000, 180000, 220000, 250000, 320000, 450000,
        500000, 750000, 1200000, 2500000, 5000000, 12000000
    ]
    value = random.choice(amounts)
    return f"{number_to_vietnamese(value)} đồng"

def time_text() -> str:
    hour = random.randint(5, 21)
    minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    if minute == 0:
        return f"{number_to_vietnamese(hour)} giờ"
    return f"{number_to_vietnamese(hour)} giờ {number_to_vietnamese(minute)} phút"

def date_text() -> str:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.choice([2024, 2025, 2026, 2027])
    return (
        f"ngày {number_to_vietnamese(day)} "
        f"tháng {number_to_vietnamese(month)} "
        f"năm {number_to_vietnamese(year)}"
    )

def phone_text() -> str:
    prefixes = [
        "032", "033", "034", "035", "036", "037", "038", "039",
        "070", "076", "077", "078", "079",
        "081", "082", "083", "084", "085", "086", "088", "089",
        "090", "091", "093", "094", "096", "097", "098", "099"
    ]
    prefix = random.choice(prefixes)
    suffix = "".join(str(random.randint(0, 9)) for _ in range(7))
    return read_digit_sequence(prefix + suffix)

def quantity_text() -> str:
    n = random.randint(2, 500)
    units = [
        "người", "cuốn sách", "tệp dữ liệu", "bức ảnh", "mẫu ghi âm",
        "tin nhắn", "đơn hàng", "bản báo cáo", "phút", "kilômét",
        "bài tập", "tài liệu", "câu thử nghiệm", "file âm thanh"
    ]
    return f"{number_to_vietnamese(n)} {random.choice(units)}"

def room_text() -> str:
    buildings = ["a", "b", "c", "d"]
    floor = random.randint(1, 9)
    room = random.randint(101, 909)
    return f"phòng {random.choice(buildings)} {read_digit_sequence(str(floor))} không {read_digit_sequence(str(room))}"

def temperature_text() -> str:
    t = random.randint(18, 38)
    return f"{number_to_vietnamese(t)} độ"

def percentage_text() -> str:
    p = random.randint(10, 95)
    return f"{number_to_vietnamese(p)} phần trăm"

def dynamic_detail() -> str:
    options = [
        f"cuộc hẹn được đặt vào lúc {time_text()}",
        f"tôi dự định bắt đầu công việc từ {time_text()}",
        f"ngày cần ghi nhớ là {date_text()}",
        f"chi phí dự kiến vào khoảng {money_text()}",
        f"số điện thoại liên hệ là {phone_text()}",
        f"khối lượng dữ liệu hiện tại khoảng {quantity_text()}",
        f"tôi đã chuẩn bị thêm khoảng {quantity_text()} cho lần thử này",
        f"thời gian hoàn thành có lẽ vào khoảng {time_text()} tối nay",
        f"nhiệt độ ngoài trời vào khoảng {temperature_text()}",
        f"mức tiến độ hiện tại vào khoảng {percentage_text()}",
        f"địa điểm gặp là {room_text()}",
    ]
    return random.choice(options)

# ==========================================
# từ vựng chung
# ==========================================

OPENERS = [
    "hôm nay",
    "sáng nay",
    "chiều nay",
    "tối nay",
    "ngày mai",
    "cuối tuần này",
    "thật ra thì hôm nay",
    "ừ thì sáng nay",
    "à hôm nay",
    "nói chung là hôm nay",
    "mấy hôm gần đây",
    "ngay từ đầu giờ sáng",
]

CONNECTORS = [
    "ngoài ra",
    "bên cạnh đó",
    "thêm vào đó",
    "vì thế",
    "do vậy",
    "nói chung",
    "thật ra",
    "thú thật",
    "đồng thời",
    "sau cùng",
    "thành ra",
    "bởi vậy",
]

CLOSERS = [
    "miễn là mọi thứ diễn ra đúng kế hoạch thì tôi sẽ yên tâm hơn nhiều",
    "chỉ cần làm cẩn thận từ đầu thì về sau sẽ đỡ mất thời gian sửa lỗi",
    "nếu hoàn thành sớm phần này thì tôi có thể dành thời gian cho việc khác",
    "nói thật là tôi muốn chuẩn bị kỹ để lúc huấn luyện đỡ gặp trục trặc",
    "hy vọng là sau vài lần thử nghiệm kết quả sẽ ổn định hơn trước",
    "làm như vậy thì dữ liệu tạo ra sẽ sạch và hữu ích hơn cho mô hình",
    "tôi nghĩ chuẩn bị kỹ như vậy sẽ giúp công việc trôi chảy hơn hẳn",
    "ít nhất thì cách làm này cũng giúp tôi kiểm soát tiến độ rõ ràng hơn",
]

SPEAKERS = {
    "sinh_vien": [
        "tôi",
        "mình",
        "tôi với bạn cùng lớp",
        "cả nhóm làm đồ án của tôi",
        "mấy bạn trong lớp",
    ],
    "van_phong": [
        "tôi",
        "chị đồng nghiệp của tôi",
        "anh trưởng nhóm",
        "người quản lý ở công ty",
        "cả phòng làm việc",
    ],
    "ban_hang": [
        "tôi",
        "chị bán hàng",
        "anh giao hàng",
        "cửa hàng bên kia",
        "bạn nhân viên tư vấn",
    ],
    "gia_dinh": [
        "mẹ tôi",
        "bố tôi",
        "chị tôi",
        "em trai tôi",
        "cả nhà tôi",
    ],
}

# ==========================================
# nội dung theo domain
# ==========================================

DOMAIN_BANK: Dict[str, Dict[str, List[str]]] = {
    "doi_song": {
        "subjects": SPEAKERS["gia_dinh"] + SPEAKERS["sinh_vien"],
        "contexts": [
            "đang dọn lại góc học tập để mọi thứ gọn gàng hơn",
            "muốn sắp xếp lại lịch sinh hoạt cho đỡ bị rối",
            "vừa chuẩn bị bữa sáng xong nên tranh thủ làm thêm ít việc",
            "đang tính ra ngoài mua vài món đồ cần thiết cho cuối tuần",
            "muốn nghỉ ngơi một chút sau mấy ngày làm việc liên tục",
            "vừa giặt quần áo xong và đang đợi trời nắng để phơi cho nhanh khô",
            "đang dọn lại bàn làm việc vì giấy tờ để lung tung quá lâu rồi",
            "muốn đổi sang một thói quen sinh hoạt đều đặn hơn để giữ sức khỏe",
        ],
        "actions": [
            "nên cần sắp xếp mọi thứ rõ ràng trước khi bắt đầu",
            "vì vậy tôi phải tranh thủ làm từng việc một cho khỏi quên",
            "nên tôi muốn ghi chú lại những đầu việc còn dang dở",
            "và tôi hy vọng chiều nay sẽ có thêm thời gian rảnh",
            "thành ra tôi đang cố gắng làm chậm mà chắc để khỏi sai sót",
            "nên tôi muốn xử lý gọn phần việc cá nhân trước khi ra ngoài",
            "vì tôi không muốn để việc nhà dồn lại quá nhiều vào cuối ngày",
        ],
        "phrases": [
            "buổi sáng tôi thường dậy sớm, uống một cốc nước rồi mới bắt đầu làm việc",
            "tối đến tôi hay nghe nhạc nhẹ rồi đọc thêm vài trang sách trước khi ngủ",
            "cuối tuần nếu rảnh tôi thường đi bộ một lúc cho đầu óc thư giãn hơn",
            "hôm qua tôi dọn lại bàn làm việc, bỏ bớt giấy tờ cũ nên thấy gọn hơn hẳn",
            "thỉnh thoảng tôi gọi điện cho bạn bè để hỏi thăm tình hình học tập và công việc",
            "mấy hôm nay thời tiết khá dễ chịu nên đi lại ngoài đường cũng đỡ mệt hơn trước",
        ],
    },
    "hoc_tap": {
        "subjects": SPEAKERS["sinh_vien"],
        "contexts": [
            "đang chuẩn bị tài liệu cho buổi học về nhận dạng tiếng nói",
            "phải hoàn thành báo cáo trước khi gửi cho giảng viên",
            "đang lên kế hoạch thu âm thêm nhiều câu dài hơn",
            "muốn luyện nói rõ ràng để bản ghi âm sạch hơn",
            "đang học cách tiền xử lý dữ liệu âm thanh",
            "vừa so sánh kết quả của hai mô hình học sâu khác nhau",
            "muốn tối ưu bộ dữ liệu để huấn luyện mô hình ổn định hơn",
            "đang ôn lại kiến thức về xác suất và học máy để theo kịp môn học",
        ],
        "actions": [
            "nên cần một không gian yên tĩnh để tập trung làm việc",
            "vì vậy tôi phải kiểm tra lại từng tệp âm thanh thật cẩn thận",
            "nên tôi dự định chia công việc thành từng bước nhỏ cho dễ theo dõi",
            "và sau đó tôi sẽ gửi toàn bộ kết quả cho nhóm cùng xem",
            "nên tôi cần thêm một ít thời gian để rà soát mọi thứ",
            "vì tôi không muốn bỏ sót những chi tiết nhỏ trong dữ liệu",
            "và có lẽ tôi sẽ thu âm lại vài câu nếu chất lượng chưa tốt",
        ],
        "phrases": [
            "tôi đang học về xử lý tiếng nói và cố gắng hiểu rõ cách xây dựng bộ dữ liệu",
            "gần đây tôi dành khá nhiều thời gian để đọc tài liệu về học máy và học sâu",
            "tôi đang thực hành huấn luyện mô hình asr tiếng việt với dữ liệu tự thu âm",
            "giảng viên yêu cầu chúng tôi phân tích kỹ chất lượng dữ liệu trước khi huấn luyện",
            "tôi muốn hiểu rõ hơn vì sao dữ liệu ít thì mô hình thường dễ bị học thuộc",
            "bây giờ tôi đang tập trung vào phần chuẩn hóa transcript để giảm lỗi khi train",
        ],
    },
    "cong_nghe_ai": {
        "subjects": SPEAKERS["sinh_vien"] + SPEAKERS["van_phong"],
        "contexts": [
            "đang thử nghiệm một mô hình trí tuệ nhân tạo mới",
            "vừa kiểm tra lại log huấn luyện của mô hình ngày hôm qua",
            "muốn đánh giá kỹ hơn chất lượng đầu ra trước khi báo cáo kết quả",
            "đang phân tích nguyên nhân vì sao mô hình hội tụ chậm hơn dự kiến",
            "vừa chạy lại thí nghiệm với cấu hình mới để so sánh sai số",
            "muốn giảm lỗi nhận dạng ở những câu dài và có nhiều số đọc",
            "đang xem lại pipeline xử lý dữ liệu từ bước tiền xử lý đến bước train",
        ],
        "actions": [
            "nên tôi phải kiểm tra kỹ từng tham số trước khi chạy lại",
            "vì thế tôi đang cố gắng nói chậm và rõ hơn bình thường",
            "và tôi hy vọng tối nay có thể hoàn thành phần việc quan trọng nhất",
            "nên tôi muốn ghi chú lại những lỗi thường gặp để sửa dần",
            "và sau đó tôi sẽ gửi toàn bộ kết quả cho nhóm cùng xem",
            "thành ra tôi cần thêm vài mẫu dữ liệu đa dạng hơn để mô hình bớt thiên lệch",
        ],
        "phrases": [
            "mô hình trí tuệ nhân tạo mạnh đến đâu thì vẫn phụ thuộc rất nhiều vào chất lượng dữ liệu đầu vào",
            "khi xây dựng bộ dữ liệu speech, tôi thấy việc đa dạng chủ đề quan trọng không kém số lượng mẫu",
            "nếu transcript và audio không khớp nhau thì mô hình nhận dạng thường học sai và khó hội tụ",
            "ngoài chuyện thu âm rõ ràng, tôi còn phải chú ý tốc độ nói và độ ổn định của âm lượng",
            "một bộ dữ liệu tốt thường có nhiều người nói, nhiều ngữ cảnh và nhiều kiểu phát âm khác nhau",
            "khi kiểm tra lỗi, tôi thường nghe lại mẫu âm thanh rồi so sánh từng từ với transcript gốc",
        ],
    },
    "lich_trinh": {
        "subjects": SPEAKERS["sinh_vien"] + SPEAKERS["van_phong"] + SPEAKERS["gia_dinh"],
        "contexts": [
            "muốn sắp xếp lại lịch làm việc để không bị trễ việc",
            "đang chuẩn bị kế hoạch cho ngày mai vì có khá nhiều việc cần xử lý",
            "phải đi qua nhiều nơi trong ngày nên muốn tính giờ thật kỹ",
            "đang kiểm tra lại danh sách cuộc hẹn để tránh bị chồng chéo",
            "muốn đổi thứ tự công việc để buổi chiều đỡ bị quá tải",
            "vừa nhận thêm một cuộc hẹn mới nên phải điều chỉnh cả lịch cũ",
            "đang cố gắng cân bằng giữa việc học, việc làm và thời gian nghỉ ngơi",
        ],
        "actions": [
            "nên tôi phải viết ra từng mốc thời gian cho dễ theo dõi",
            "vì vậy tôi muốn xác nhận lại địa điểm và giờ giấc thật cụ thể",
            "và tôi hy vọng mọi thứ sẽ kịp trước giờ đã hẹn",
            "nên tôi cần chuẩn bị sẵn giấy tờ từ bây giờ cho chắc",
            "thành ra tôi đang ưu tiên những việc quan trọng hơn trước",
            "vì tôi không muốn phải thay đổi lịch vào phút cuối",
        ],
        "phrases": [
            "chiều nay tôi sẽ đến trường sớm hơn một chút để kịp chuẩn bị cho buổi thảo luận",
            "ngày mai tôi có khá nhiều việc nên phải sắp xếp thứ tự ưu tiên ngay từ bây giờ",
            "cuối tuần này tôi dự định dành một buổi để hoàn thành những phần còn dang dở",
            "nếu xong việc sớm thì tôi sẽ tranh thủ ghé qua thư viện để mượn thêm tài liệu",
            "tôi đang xem lại lịch để tránh trùng giờ giữa buổi học và cuộc hẹn bên ngoài",
            "mỗi khi có nhiều việc cùng lúc tôi thường ghi ra giấy để theo dõi cho đỡ sót",
        ],
    },
    "chi_dan_yeu_cau": {
        "subjects": SPEAKERS["van_phong"] + SPEAKERS["sinh_vien"],
        "contexts": [
            "đang cần người hỗ trợ một số việc nhỏ trong lúc chuẩn bị",
            "muốn kiểm tra lại thiết bị trước khi bắt đầu ghi âm",
            "đang cố gắng sắp xếp tài liệu cho gọn để cả nhóm dễ theo dõi",
            "muốn giảm bớt tiếng ồn trong phòng để chất lượng âm thanh tốt hơn",
            "đang cần xác nhận lại danh sách file để tránh thiếu dữ liệu",
            "muốn nhờ người khác rà soát giúp những chỗ tôi chưa chắc",
        ],
        "actions": [
            "nên tôi phải nhờ thêm một người hỗ trợ cho nhanh",
            "vì vậy tôi muốn mọi thứ được chuẩn bị rõ ràng ngay từ đầu",
            "thành ra tôi cần hướng dẫn cụ thể để tránh nhầm lẫn",
            "nên tôi muốn xác nhận lại từng bước trước khi làm tiếp",
            "và như vậy thì công việc của cả nhóm sẽ trôi chảy hơn",
        ],
        "phrases": [
            "bạn làm ơn mở cửa sổ giúp tôi một chút vì trong phòng hơi bí",
            "nếu tiện thì bạn gửi lại cho tôi bản tài liệu mới nhất qua email nhé",
            "lát nữa bạn nhắc tôi kiểm tra micro trước khi bắt đầu ghi âm được không",
            "bạn nhớ giảm bớt tiếng quạt trong phòng để tệp thu âm đỡ bị nhiễu nhé",
            "khi nào rảnh bạn xem giúp tôi danh sách câu đọc còn thiếu để tôi bổ sung",
            "nếu có thể thì bạn sắp xếp lại tên file theo đúng thứ tự cho dễ quản lý",
        ],
    },
    "mua_sam_thanh_toan": {
        "subjects": SPEAKERS["ban_hang"] + SPEAKERS["gia_dinh"] + ["tôi"],
        "contexts": [
            "đang cân nhắc mua thêm vài món đồ cần thiết cho công việc",
            "muốn so sánh giá trước khi quyết định đặt hàng",
            "vừa kiểm tra giỏ hàng và thấy còn thiếu vài món quan trọng",
            "đang tính lại chi phí để không vượt quá số tiền đã dự trù",
            "muốn hỏi kỹ về thời gian giao hàng trước khi thanh toán",
            "đang xem lại thông tin đơn hàng để tránh nhập sai địa chỉ",
        ],
        "actions": [
            "nên tôi phải hỏi kỹ từng chi tiết trước khi chốt đơn",
            "vì vậy tôi muốn kiểm tra lại giá và phí vận chuyển",
            "thành ra tôi đang so sánh giữa hai cửa hàng khác nhau",
            "nên tôi muốn chắc chắn là sản phẩm đúng với nhu cầu sử dụng",
            "và như vậy thì tôi sẽ yên tâm hơn khi thanh toán",
        ],
        "phrases": [
            "hôm qua tôi vào cửa hàng để xem thử một vài mẫu tai nghe phục vụ cho việc học",
            "trước khi đặt mua thiết bị mới tôi thường đọc kỹ phần mô tả và đánh giá của người dùng",
            "nếu mức giá hợp lý và thời gian giao hàng nhanh thì tôi sẽ cân nhắc chốt đơn ngay",
            "có những lúc tôi phải tính rất kỹ vì tổng chi phí phát sinh cao hơn dự kiến ban đầu",
            "để tránh nhầm lẫn tôi thường kiểm tra lại số lượng, màu sắc và địa chỉ nhận hàng",
            "nhiều khi sản phẩm nhìn khá ổn nhưng phí giao hàng cộng vào lại làm giá tăng lên đáng kể",
        ],
    },
    "goi_dien_dat_lich": {
        "subjects": ["tôi", "bên lễ tân", "chị tư vấn", "anh phụ trách", "bạn ở đầu dây bên kia"],
        "contexts": [
            "đang gọi điện để xác nhận lại cuộc hẹn đã đặt trước đó",
            "muốn dời lịch sang một khung giờ phù hợp hơn",
            "đang hỏi thêm thông tin về địa điểm và thủ tục cần chuẩn bị",
            "muốn kiểm tra xem phía bên kia còn chỗ trống trong ngày hay không",
            "đang ghi lại thông tin để tránh quên khi đến nơi",
            "muốn xác nhận lại số điện thoại liên hệ trước khi kết thúc cuộc gọi",
        ],
        "actions": [
            "nên tôi nói chậm từng ý để hai bên dễ hiểu hơn",
            "vì vậy tôi phải nhắc lại thông tin quan trọng một lần nữa",
            "thành ra tôi đang ghi chú ngay trong lúc nói chuyện",
            "nên tôi muốn xác nhận lại giờ giấc và địa chỉ thật rõ ràng",
            "và như vậy thì khi đến nơi tôi sẽ không bị lúng túng",
        ],
        "phrases": [
            "khi gọi điện đặt lịch tôi thường chuẩn bị sẵn nội dung cần hỏi để cuộc trò chuyện ngắn gọn hơn",
            "nếu bên kia nói nhanh quá thì tôi sẽ xin họ nhắc lại từng thông tin một cho chắc",
            "nhiều lúc chỉ cần nhầm một con số trong số điện thoại là việc liên hệ sau đó rất phiền",
            "trước khi kết thúc cuộc gọi tôi luôn kiểm tra lại ngày, giờ và địa điểm để tránh sai sót",
            "nếu lịch thay đổi vào phút cuối thì tôi cũng muốn được báo sớm để còn chủ động sắp xếp",
            "đôi khi tôi phải gọi lại lần hai chỉ để xác nhận xem cuộc hẹn đã được ghi nhận hay chưa",
        ],
    },
    "ke_chuyen_mo_ta": {
        "subjects": SPEAKERS["sinh_vien"] + SPEAKERS["gia_dinh"] + SPEAKERS["van_phong"],
        "contexts": [
            "đang nhớ lại một tình huống khá thú vị xảy ra vào tuần trước",
            "vừa kể cho bạn nghe một câu chuyện nhỏ trong lúc đi học",
            "muốn mô tả rõ hơn việc đã xảy ra để mọi người dễ hình dung",
            "đang nhắc lại một lần đi nhầm địa điểm vì đọc thiếu thông tin",
            "vừa kể về lần thu âm bị hỏng do tiếng ồn bất ngờ xuất hiện",
            "đang nhớ tới một buổi làm việc kéo dài đến tối muộn mới xong",
        ],
        "actions": [
            "nên đến bây giờ nghĩ lại tôi vẫn thấy khá buồn cười",
            "vì vậy tôi rút ra được vài kinh nghiệm cho những lần sau",
            "thành ra sau chuyện đó tôi cẩn thận hơn rất nhiều",
            "nên từ đó tôi luôn kiểm tra lại thông tin trước khi bắt đầu",
            "và thật ra nhờ vậy mà lần sau mọi thứ ổn hơn hẳn",
            "nên tôi muốn kể kỹ một chút để người khác tránh gặp lỗi tương tự",
        ],
        "phrases": [
            "hôm trước tôi đi xe buýt đến trường, giữa đường trời đổ mưa nên ai cũng vội vàng tìm chỗ trú",
            "có lần tôi đang ghi âm thì điện thoại reo liên tục, thế là phải thu lại gần như từ đầu",
            "hôm qua tôi gặp một bạn cùng ngành, hai đứa ngồi nói chuyện khá lâu về mô hình ngôn ngữ",
            "một lần khác tôi lưu nhầm file vào thư mục cũ nên tìm mãi mới thấy dữ liệu cần dùng",
            "đợt trước nhóm tôi phải làm việc đến tối muộn để kịp nộp báo cáo đúng hạn cho thầy",
            "có hôm tôi đọc câu quá nhanh, nghe lại mới thấy nhiều chỗ bị nuốt âm nên phải sửa ngay",
        ],
    },
}

# ==========================================
# kiểu câu theo speaker style
# ==========================================

SPEAKER_FILLERS = {
    "sinh_vien": ["thật ra", "kiểu như", "nói chung là", "à thì", "ừm"],
    "van_phong": ["thực ra", "nói chung", "về cơ bản", "thành ra", "nhìn chung"],
    "ban_hang": ["nói thật", "thực ra", "nếu vậy", "thành ra", "bên mình thì"],
    "gia_dinh": ["ờ", "à", "thật ra", "nói chung", "vậy nên"],
}

def infer_style(subject: str) -> str:
    for style, items in SPEAKERS.items():
        if subject in items:
            return style
    return "sinh_vien"

def maybe_add_filler(sentence: str, style: str) -> str:
    if random.random() < 0.35:
        filler = random.choice(SPEAKER_FILLERS[style])
        return f"{filler}, {sentence[0].lower()}{sentence[1:]}"
    return sentence

# ==========================================
# template sinh câu
# ==========================================

def normalize_spaces(text: str) -> str:
    return " ".join(text.split())

def word_count(text: str) -> int:
    clean = text.replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ")
    return len(clean.split())

def choose_subject(domain: str) -> str:
    return random.choice(DOMAIN_BANK[domain]["subjects"])

def choose_parts(domain: str) -> Tuple[str, str, str]:
    bank = DOMAIN_BANK[domain]
    return (
        random.choice(bank["phrases"]),
        random.choice(bank["contexts"]),
        random.choice(bank["actions"]),
    )

def build_sentence(domain: str) -> str:
    subject = choose_subject(domain)
    phrase, context, action = choose_parts(domain)
    style = infer_style(subject)

    templates = [
        f"{random.choice(OPENERS)}, {subject} {context}, {action}, {random.choice(CONNECTORS)} {dynamic_detail()}, {random.choice(CLOSERS)}.",
        f"{phrase}, {random.choice(CONNECTORS)} {subject} {context}, {dynamic_detail()}, {random.choice(CLOSERS)}.",
        f"{random.choice(OPENERS)}, {phrase}, vì {subject} {context}, {action}, {dynamic_detail()}.",
        f"{subject} {context}, {action}, {random.choice(CONNECTORS)} {phrase}, {dynamic_detail()}, {random.choice(CLOSERS)}.",
        f"{phrase}, lúc đó {subject} {context}, {action}, {random.choice(CONNECTORS)} {dynamic_detail()}, {random.choice(CLOSERS)}.",
        f"{random.choice(OPENERS)}, {subject} {context}, {random.choice(CONNECTORS)} {phrase}, {action}, {random.choice(CLOSERS)}.",
    ]

    sentence = random.choice(templates)
    sentence = normalize_spaces(sentence)
    sentence = maybe_add_filler(sentence, style)
    return sentence

def make_sentence(domain: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> str:
    for _ in range(300):
        sentence = build_sentence(domain)
        wc = word_count(sentence)
        if min_words <= wc <= max_words:
            return sentence

    # fallback dài hơn nếu thiếu từ
    subject = choose_subject(domain)
    phrase, context, action = choose_parts(domain)
    fallback = (
        f"{random.choice(OPENERS)}, {subject} {context}, {action}, "
        f"{random.choice(CONNECTORS)} {phrase}, {dynamic_detail()}, {random.choice(CLOSERS)}."
    )
    return normalize_spaces(fallback)

# ==========================================
# chống trùng gần giống
# ==========================================

def canonical_text(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("?", "")
        .replace("!", "")
        .strip()
    )

def shingle_signature(text: str, k: int = 4) -> Tuple[str, ...]:
    words = canonical_text(text).split()
    if len(words) < k:
        return tuple(words)
    return tuple(" ".join(words[i:i + k]) for i in range(len(words) - k + 1))

def too_similar(candidate: str, existing_signatures: List[Tuple[str, ...]]) -> bool:
    cand_sig = set(shingle_signature(candidate))
    if not cand_sig:
        return False

    # chỉ so với một phần cuối để tăng tốc
    for sig in existing_signatures[-120:]:
        sig_set = set(sig)
        overlap = len(cand_sig & sig_set)
        ratio = overlap / max(1, min(len(cand_sig), len(sig_set)))
        if ratio >= 0.7:
            return True
    return False

# ==========================================
# sinh dataset cân bằng domain
# ==========================================

def generate_dataset(
    output_txt: str = OUTPUT_FILE,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> None:
    sentences: List[str] = []
    seen_exact = set()
    seen_signatures: List[Tuple[str, ...]] = []

    for domain, n_items in DOMAIN_COUNTS.items():
        count = 0
        attempts = 0
        max_attempts = n_items * 200

        while count < n_items and attempts < max_attempts:
            attempts += 1
            sentence = make_sentence(domain, min_words=min_words, max_words=max_words)
            key = canonical_text(sentence)

            if key in seen_exact:
                continue
            if too_similar(sentence, seen_signatures):
                continue

            seen_exact.add(key)
            seen_signatures.append(shingle_signature(sentence))
            sentences.append(sentence)
            count += 1

        if count < n_items:
            raise RuntimeError(
                f"domain {domain} chỉ sinh được {count}/{n_items} câu. "
                "hãy tăng thêm phrase hoặc template cho domain đó."
            )

    random.shuffle(sentences)
    Path(output_txt).write_text("\n".join(sentences), encoding="utf-8")

    print(f"đã tạo {len(sentences)} câu vào file {output_txt}")
    print("phân bổ domain:")
    for domain, n_items in DOMAIN_COUNTS.items():
        print(f"  {domain}: {n_items}")

# ==========================================
# chạy
# ==========================================

if __name__ == "__main__":
    generate_dataset()