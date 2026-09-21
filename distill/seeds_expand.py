import json
import random
from pathlib import Path

random.seed(42)

PRODUCTS = ["同步盘", "协作平台", "API 网关", "数据看板", "移动 APP", "桌面客户端", "在线文档", "日历服务"]
PLANS = ["免费版", "标准版", "年度套餐", "企业版", "试用版", "教育版"]
URGENT_AILMENTS = [
    "完全打不开", "一直转圈加载", "闪退", "卡在登录页", "数据全部丢失", "账号被锁死",
    "文件全部损坏", "同步彻底停了", "白屏", "报错 500",
]
MILD_AILMENTS = [
    "偶尔卡顿", "界面字体显示异常", "通知延迟", "搜索速度慢", "导出格式有点乱",
    "图标显示不出来", "快捷键冲突", "列表滚动不流畅", "夜间模式颜色刺眼",
]
BILLING_EVENTS = [
    "被重复扣款", "扣款金额不对", "退款一直没到账", "自动续费没经我同意", "升级后多收了差价",
    "优惠券没有生效", "账单金额和订单不一致", "免费试用期就开始扣费",
]
POLITE = ["麻烦了", "辛苦帮忙看下", "谢谢", "拜托了", "麻烦尽快", ""]
ANGRY = ["太让人失望了", "再不解决就退订", "服务也太差了吧", "已经投诉了", "最后一次机会", "无语了"]
BUGS = [
    "上传到 90% 就失败", "共享链接打不开", "批量下载缺文件", "搜索结果不完整",
    "回收站清空后文件找不回", "版本历史只显示一半", "权限设置不生效", "审计日志缺失",
]
FEATURES = [
    "希望支持批量重命名", "想要离线编辑", "希望能和飞书集成", "需要审计日志导出",
    "建议增加模板中心", "希望支持二次验证", "想要更细的权限粒度", "希望有客户端桌面小组件",
]
ECOM_ORDER_IDS = [f"SO{random.randint(100000, 999999)}" for _ in range(40)]
ECOM_PRODUCTS = ["无线耳机", "机械键盘", "显示器支架", "充电宝", "蓝牙音箱", "摄像头", "路由器"]
ECOM_ISSUES = [
    ("物流", "物流卡在{city}一周了，订单{oid}一直没有更新，客服电话也打不通。", ["上海", "深圳", "武汉", "成都"]),
    ("物流", "下单五天了还没发货，{oid}显示揽收后没有下文，请尽快处理。", [""]),
    ("商品", "收到的{prod}有明显的使用痕迹，和描述不符，要求退换货。", [""]),
    ("商品", "{prod}用了三天就坏了，开不了机，这是质量问题要求换新。", [""]),
    ("退款", "七天无理由退货申请被驳回，{oid}商品没有任何问题，请重新审核。", [""]),
    ("退款", "退货已经寄回十天了，退款还没有到账，订单{oid}，请尽快处理。", [""]),
    ("价格", "昨天买完今天就降价了，{prod}差价 {n} 元，要求补差价。", [""]),
    ("服务", "客服让我等 48 小时，问题一点进展都没有，订单{oid}。", [""]),
]
INC_SERVICES = ["支付网关", "用户中心", "对象存储", "消息队列", "报表服务", "登录服务", "短信网关"]
INC_LEVELS = [
    ("P1", "生产事故：{svc}不可用，影响{env}环境，请立即响应。", ["全部", "华东", "核心"]),
    ("P2", "{svc}部分功能异常，{env}受影响，需要排查。", ["部分用户", "灰度"]),
    ("P3", "{svc}告警{alert}，尚未确认影响面。", ["延迟升高", "错误率上升", "磁盘水位 85%"]),
]
REVIEW_STAR_TEXTS = [
    (5, "很好用，{p}体验流畅，团队都在用了。"),
    (5, "功能齐全，客服响应也快，好评。"),
    (4, "整体不错，就是偶尔同步慢一点。"),
    (3, "功能还行，但{p}经常小毛病，希望改进。"),
    (2, "{p}{a}，体验很差，差评。"),
    (1, "垃圾软件，{a}不说，客服还爱搭不理，要求退款。"),
]
EMAIL_INTERNAL = [
    "各位，本周五晚 {h} 点到次日 6 点{svc}进行升级维护，期间服务可能中断，请提前安排。",
    "提醒：本季度{p}的企业版合同 {m} 月 {d} 日到期，续约请在本周内确认价格。",
    "新同事入职流程更新：账号开通需提前 {d} 个工作日提交申请，附部门与岗位信息。",
    "关于上季度客诉汇总：共 {n} 起，其中账单类 {b} 起、故障类 {t} 起，详见附件。",
    "预算审批：{p}的采购费用 {n} 元已通过，下周走付款流程。",
]
MIXED_SIGNALS = [
    "再不解决{p}同步失败的问题我就退订了，钱我也不想要了，太坑了。",
    "重复扣款两次了我已经申请退款，顺便说下{p}还老闪退，心累。",
    "{p}传文件失败率 50%，我要退订，把没用的天数退钱给我。",
    "你们上次的处理我很不满意，本来要给差评的，看在新版本还行的份上再观察一段时间。",
    "已经向 12315 投诉你们乱扣费了，除非今天内给我退款电话。",
    "我们公司在做续约评估，但{p}这个季度的稳定性实在让人犹豫。",
    "用了一年本来挺满意，这次的数据丢失让我开始考虑别的产品了。",
]
EN_TEMPLATES = [
    ("tech", "The {p} has been down for {n} hours, our team is completely blocked."),
    ("tech", "App crashes on startup since the last update, tried reinstalling {n} times."),
    ("tech", "Export to CSV keeps failing at {n}0%, please investigate."),
    ("billing", "I was double charged for my {m} invoice, please refund the duplicate."),
    ("billing", "Your pricing page says ${n} per user but we were billed ${n2}. Explain."),
    ("billing", "Cancel the auto-renewal immediately, I never agreed to it."),
    ("sales", "Can we get a quote for {n} seats with SSO and audit logs?"),
    ("sales", "Is there a nonprofit discount? We are a {n}-person organization."),
    ("sales", "We are evaluating vendors, can you arrange a demo this week?"),
    ("churn", "Cancel my subscription immediately, the service quality dropped badly."),
    ("churn", "If this issue is not fixed by Friday we are migrating to a competitor."),
    ("praise", "Love the new release, the {p} feels much faster now. Great work."),
    ("question", "How do I transfer admin rights to another teammate?"),
    ("question", "Does the {p} support SSO with our identity provider?"),
]

rows = []
n = 0


def add(text: str, tag: str):
    global n
    n += 1
    holdout = random.random() < 0.05
    rows.append({"id": f"v3-{n:04d}", "text": text, "tag": tag, "holdout": holdout})


# ---- A. SaaS 工单 ----
for p in PRODUCTS:
    for a in URGENT_AILMENTS:
        tone = random.choice(ANGRY + POLITE)
        add(f"{p}{a}了，{'版本 4.{v}.1，' if (v := random.randint(0, 9)) is not None else ''}{tone}".strip(), "tech_urgent")
    for a in random.sample(MILD_AILMENTS, 4):
        add(f"反馈一个问题：{p}{a}，不影响主流程但希望优化一下。", "tech_mild")
    for b in random.sample(BUGS, 3):
        add(f"{p}{b}，复现步骤已经写在附件里，麻烦看下。", "tech_bug")
    for f in random.sample(FEATURES, 3):
        add(f"建议：{f}，我们团队经常需要这个能力。", "feature")
    for e in random.sample(BILLING_EVENTS, 3):
        plan = random.choice(PLANS)
        add(f"我在{plan}的{p}上{e}，请核对账单并处理，发票信息可以看我资料。", "billing")
    add(f"想了解{p}企业版的报价，我们大概 {random.randint(50, 800)} 人规模。", "sales")
    add(f"请问{p}有教育优惠吗？我们是{random.choice(['学校实验室', '培训机构', '公益组织'])}。", "sales")

# ---- B. 电商 ----
for cat, tpl, pools in ECOM_ISSUES:
    for _ in range(random.randint(6, 10)):
        prod = random.choice(ECOM_PRODUCTS)
        city = random.choice(pools) if pools else ""
        text = tpl.format(prod=prod, oid=random.choice(ECOM_ORDER_IDS), city=city,
                          n=random.choice([30, 50, 80, 120, 200]))
        add(text, f"ecom_{cat}")

# ---- C. IT 事故 ----
for svc in INC_SERVICES:
    for lvl, tpl, envs in INC_LEVELS:
        add(tpl.format(svc=svc, env=random.choice(envs),
                       alert=random.choice(["延迟升高", "错误率上升", "连接数异常", "CPU 90%"])),
            f"incident_{lvl}")

# ---- D. 应用商店评价 ----
for p in PRODUCTS[:4]:
    for star, tpl in REVIEW_STAR_TEXTS:
        for _ in range(random.randint(2, 4)):
            a = random.choice(URGENT_AILMENTS + MILD_AILMENTS)
            add(tpl.format(p=p, a=a), f"review_{star}star")

# ---- E. 内部邮件 ----
for tpl in EMAIL_INTERNAL:
    for _ in range(random.randint(3, 5)):
        add(tpl.format(svc=random.choice(INC_SERVICES), p=random.choice(PRODUCTS),
                       h=random.choice([22, 23, 0]), m=random.randint(1, 12), d=random.randint(1, 28),
                       n=random.randint(3, 40), b=random.randint(1, 15), t=random.randint(1, 20)),
            "internal")

# ---- F. 混合信号 ----
for p in PRODUCTS:
    for tpl in MIXED_SIGNALS[:3]:
        add(tpl.format(p=p), "mixed")
add(MIXED_SIGNALS[3], "mixed_soft")
add(MIXED_SIGNALS[4], "mixed_legal")
add(MIXED_SIGNALS[5], "mixed_renewal")
add(MIXED_SIGNALS[6], "mixed_reconsider")

# ---- G. 英文 ----
EN_PRODUCTS = ["sync drive", "dashboard", "mobile app", "admin console"]
for kind, tpl in EN_TEMPLATES:
    for _ in range(random.randint(5, 8)):
        p = random.choice(EN_PRODUCTS)
        m = random.choice(["March", "April", "May"])
        text = tpl.format(p=p, n=random.randint(2, 24), n2=random.randint(50, 90),
                          m=m)
        add(text, f"en_{kind}")

random.shuffle(rows)

# ---- 变体扩展：加语境/时间/设备前缀与语气后缀，扩充到目标规模 ----
PREFIXES = [
    "你好，", "您好，", "管理员你好，", "", "", "",
    "从上周开始，", "最近几天，", "从昨天开始，", "升级之后，",
    "在鸿蒙平板上，", "在 iOS 上，", "在网页端，", "公司网络环境下，",
]
SUFFIXES = [
    "，已经两天了。", "，请尽快回复。", "，在线等。", "，谢谢。", "。", "", "",
    "，附了截图。", "，麻烦了。", "，等你们消息。",
]
TARGET = 1700
base = list(rows)
while len(rows) < TARGET and base:
    src = random.choice(base)
    text = src["text"]
    if len(text) + 30 > 900:
        continue
    pre = random.choice(PREFIXES)
    suf = random.choice(SUFFIXES)
    variant = f"{pre}{text}{suf}".replace("。。", "。").replace("，，", "，")
    if variant == text or any(r["text"] == variant for r in rows):
        continue
    n += 1
    rows.append({"id": f"v3-{n:04d}", "text": variant, "tag": src["tag"],
                 "holdout": random.random() < 0.05})

random.shuffle(rows)
out = Path(__file__).parent / "seeds_v3.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
hold = sum(1 for r in rows if r["holdout"])
print(f"wrote {len(rows)} seeds ({hold} holdout) -> {out}")
