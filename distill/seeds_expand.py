import json
import random
from pathlib import Path

random.seed(42)

PRODUCTS = ["同步盘", "协作平台", "API 网关", "数据看板", "移动 APP", "桌面客户端"]
PLANS = ["免费版", "标准版", "年度套餐", "企业版", "试用版"]
URGENT_AILMENTS = [
    "完全打不开", "一直转圈加载", "闪退", "卡在登录页", "数据全部丢失", "账号被锁死",
]
MILD_AILMENTS = [
    "偶尔卡顿", "界面字体显示异常", "通知延迟", "搜索速度慢", "导出格式有点乱",
]
BILLING_EVENTS = ["被重复扣款", "扣款金额不对", "退款一直没到账", "自动续费没经我同意", "升级后多收了差价"]
POLITE = ["麻烦了", "辛苦帮忙看下", "谢谢", "拜托了", ""]
ANGRY = ["太让人失望了", "再不解决就退订", "服务也太差了吧", "已经投诉了", "最后一次机会"]

rows = []
n = 0


def add(text: str, tag: str):
    global n
    n += 1
    rows.append({"id": f"v2-{n:04d}", "text": text, "tag": tag})


# 1. 技术故障（紧急 + 轻微）
for i, p in enumerate(PRODUCTS):
    for j, a in enumerate(URGENT_AILMENTS):
        tone = random.choice(ANGRY + POLITE)
        add(f"{p}{a}了，{'版本是最新的，' if random.random() < 0.5 else ''}{tone}".strip(), "tech_urgent")
for p in PRODUCTS:
    for a in random.sample(MILD_AILMENTS, 3):
        add(f"反馈一个问题：{p}{a}，不影响主流程但希望优化一下。", "tech_mild")

# 2. 账单类
for p in PRODUCTS:
    for e in BILLING_EVENTS:
        for plan in random.sample(PLANS, 2):
            add(f"我在{plan}的{p}上{e}，请核对账单并处理，发票信息可以看我资料。", "billing")
add("能不能把这个月的账单明细发我一份？想核对一下用量计费。", "billing")
add("对公转账的银行信息在哪里能找到？财务需要打款。", "billing")
add("月底了要报销，能不能补一份盖章版发票？", "billing")
add("你们的定价页写着每用户每月 45，为什么账单上是 60？", "billing")

# 3. 销售/咨询
for p in PRODUCTS:
    add(f"想了解{p}企业版的报价，我们大概 {random.randint(50, 800)} 人规模。", "sales")
    add(f"请问{p}有教育优惠吗？我们是学校实验室。", "sales")
    add(f"希望能安排一次{p}的演示，最好这周。", "sales")
add("续费有优惠吗？老用户了。", "sales")
add("你们和 A 产品比有什么优势？正在选型。", "sales")
add("支持私有化部署吗？报价怎么算？", "sales")

# 4. 其他/反馈/咨询
for p in PRODUCTS:
    add(f"{p}的深色模式不错，建议加个自动跟随系统的开关。", "feedback")
    add(f"请问{p}什么时候支持多语言界面？", "question")
add("感谢团队上次的快速修复，体验越来越好了。", "praise")
add("如何把旧账号的数据合并到新账号？", "question")
add("你们招聘吗？在哪里投简历？", "other")
add("我觉得你们的 logo 换了之后不如以前好看。", "feedback")

# 5. 混合信号（取消威胁 + 技术问题；退款 + 抱怨）——教师易错区
add("再不解决同步失败的问题我就退订了，钱我也不想要了，太坑了。", "tech+churn")
add("重复扣款两次了我已经申请退款，顺便说下 APP 还老闪退，心累。", "billing+tech")
add("网盘传文件失败率 50%，我要退订，把没用的天数退钱给我。", "tech+churn+refund")

# 6. 英文
EN = [
    ("The dashboard has been down for two hours, our ops team is blocked.", "tech_urgent"),
    ("I was double charged this month, please refund the duplicate.", "billing"),
    ("Can we get a quote for 200 seats with SSO?", "sales"),
    ("Love the new release, could you add dark mode scheduling?", "feedback"),
    ("Export to CSV keeps failing at 90%, please investigate.", "tech_mild"),
    ("Cancel my subscription immediately, the service quality dropped.", "churn"),
    ("Invoice for March shows wrong VAT number, can you reissue?", "billing"),
    ("Is there a nonprofit discount available?", "sales"),
]
for t, tag in EN:
    add(t, tag)

random.shuffle(rows)
out = Path(__file__).parent / "seeds_v2.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"wrote {len(rows)} seeds -> {out}")
