import json
import random
from pathlib import Path

random.seed(4242)

# v4: 扩域加练 —— 教育/医疗/金融/IT 运维/内容审核 + 新形态（多轮对话/JSON/长邮件）
rows = []
n = 0


def add(text, tag):
    global n
    n += 1
    rows.append({"id": f"v4-{n:04d}", "text": text, "tag": tag, "holdout": random.random() < 0.05})


EDU = [("网课平台", ["直播卡顿掉线", "作业提交后显示未交", "回放视频黑屏", "题库答案加载失败"]),
       ("在线题库", ["刷题进度不同步", "错题本清空", "模拟考交卷失败"])]
MED = [("预约系统", ["挂号一直排队中", "报告查询白屏", "缴费后订单消失"]),
       ("问诊 APP", ["医生回复延迟两天", "处方单打不开", "药品配送地址改不了"])]
FIN = [("记账应用", ["账单分类丢失", "同步后余额不对", "导出流水乱码"]),
       ("支付网关", ["回调丢失导致掉单", "手续费计算与合同不符", "退款原路退回失败"])]
OPS = [("监控系统", ["告警风暴凌晨触发", "大盘图表不刷新", "阈值配置不生效"]),
       ("部署平台", ["灰度发布卡在 50%", "回滚按钮无响应", "构建日志丢失"])]
MOD = [("内容平台", ["评论被误删", "敏感词拦截过严", "举报三小时无人处理"])]

for domain_set, tag_prefix in [(EDU, "edu"), (MED, "med"), (FIN, "fin"), (OPS, "ops"), (MOD, "mod")]:
    for product, ailments in domain_set:
        for a in ailments:
            tone = random.choice(["麻烦尽快处理。", "已经影响正常使用了。", "再这样只能换别家了。", "", "谢谢。"])
            add(f"{product}{a}，{tone}".strip(), f"{tag_prefix}_tech")
        add(f"{product}的企业版怎么收费？我们 {random.randint(30, 500)} 人。", f"{tag_prefix}_sales")
        add(f"{product}上个月的账单好像多扣了一笔，帮忙核对下。", f"{tag_prefix}_billing")

# 多轮对话形态
DIALOGS = [
    "用户：退款怎么还没到？\n客服：已加急，1-3 个工作日。\n用户：都第五天了！再不到我就投诉了。",
    "用户：APP 又闪退了。\n客服：请提供机型和版本。\n用户：安卓 14，昨天更新的 3.2.1，每次打开都退。",
    "用户：发票抬头能改吗？\n客服：可以，提供新抬头。\n用户：好，改成 XX 科技有限公司，税号 91xxxx。",
]
for d in DIALOGS:
    add(d, "dialog")

# JSON 工单形态
for _ in range(8):
    add(json.dumps({
        "severity": random.choice(["P1", "P2", "P3"]),
        "service": random.choice(["登录服务", "支付回调", "消息推送", "对象存储"]),
        "error_rate": f"{random.randint(2, 40)}%",
        "since": f"{random.randint(1, 12)}h ago",
        "customer_tier": random.choice(["free", "standard", "enterprise"]),
    }, ensure_ascii=False), "json_state")

# 长邮件形态
for _ in range(10):
    add(f"""您好：

我们团队使用贵方产品已有一年，整体体验尚可。但最近一个月遇到几个问题：
1. 同步延迟明显增加，早高峰时段尤为严重；
2. 上周有一次服务中断约 {random.randint(10, 90)} 分钟，未见任何公告；
3. 客服响应时间从原来的 2 小时延长到超过 24 小时。

我们正在评估年度续约，希望贵方重视上述问题，并在本周内给出书面答复。

此致
{random.choice(['王经理', '刘工', '张总监'])}""", "long_email")

random.shuffle(rows)

# 变体扩展（同 v3 手法）：语境前缀 + 语气后缀，扩到目标规模
PREFIXES = ["你好，", "您好，", "客服你好，", "", "", "", "从昨天起，", "最近，", "升级后，", "手机端，"]
SUFFIXES = ["，请尽快回复。", "，在线等。", "，谢谢。", "。", "", "", "，麻烦了。", "，急。"]
TARGET = 160
base = list(rows)
while len(rows) < TARGET and base:
    src = random.choice(base)
    text = src["text"]
    if len(text) + 30 > 900:
        continue
    variant = f"{random.choice(PREFIXES)}{text}{random.choice(SUFFIXES)}".replace("。。", "。").replace("，，", "，")
    if variant == text or any(r["text"] == variant for r in rows):
        continue
    n += 1
    rows.append({"id": f"v4-{n:04d}", "text": variant, "tag": src["tag"],
                 "holdout": random.random() < 0.05})

random.shuffle(rows)
out = Path(__file__).parent / "seeds_v4.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
hold = sum(1 for r in rows if r["holdout"])
print(f"wrote {len(rows)} seeds ({hold} holdout) -> {out}")
