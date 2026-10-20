"""Rebuild the versioned, synthetic retrieval fixture. No LLM-generated labels at runtime."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each tuple: document key/title, two exact evidence passages and their question variants.
# These are fictional policies, not statements of employment law or real company rules.
POLICIES = [
    ("travel-budget", "星桥科技差旅费用标准（合成）", [
        ("hotel", "境内出差住宿按城市分档：杭州、南京每人每晚550元，其他城市每人每晚400元。",
         "到杭州出差的住宿报销上限是多少？", "去杭州办事住酒店，一晚上最多能报多少钱？",
         "那南京每晚最多多少？"),
        ("train", "境内高铁默认购买二等座；二等座售罄且任务紧急时，购买一等座前须取得部门负责人书面批准。",
         "高铁二等座售罄时能否购买一等座？", "出差的普通高铁座位卖完了，升级座位要谁同意？",
         "那升级一等座需要什么条件？"),
    ]),
    ("travel-process", "星桥科技出差申请与报销流程（合成）", [
        ("approval", "出差前至少两个工作日提交申请，依次由直属主管和部门负责人审批；涉及跨境出差时还需行政负责人复核。",
         "国内出差需要提前多久申请，由谁审批？", "下周去外地工作，申请应该提前几天交给哪些人？",
         "那跨境出差还要谁复核？"),
        ("deadline", "差旅结束后十个工作日内提交报销单；因出差期间系统故障无法提交的，应附故障工单并在恢复后三个工作日内补交。",
         "差旅结束后的报销提交期限是什么？", "从外地回来以后，最迟什么时候把报销材料交上去？",
         "那系统故障导致没交上怎么办？"),
    ]),
    ("invoice", "星桥科技票据与付款核验细则（合成）", [
        ("receipt", "费用报销应同时提供合规发票、付款凭证和对应审批单；住宿还需提供注明入住及离店日期的酒店水单。",
         "住宿报销必须准备哪些材料？", "住店后申请报账，除了发票还需要交什么？",
         "那酒店水单需要包含哪些日期？"),
        ("lost-invoice", "电子发票丢失应先向开票方申请重发；确实无法补开的，不得用其他消费发票替代，须提交异常说明交财务审核。",
         "电子发票丢失且无法补开时如何处理？", "付款后票据找不到了，可以拿别的消费票据顶替吗？",
         "那申请重发也办不了时怎么办？"),
    ]),
    ("purchase", "星桥科技采购申请管理规则（合成）", [
        ("quotes", "单笔采购预算达到5000元时，至少取得三家供应商报价并填写比价记录；低于5000元的采购仍需主管批准。",
         "采购金额达到多少时需要三家报价？", "想买一批总价六千元的设备，需要找几家商家比价？",
         "那不足五千元还要审批吗？"),
        ("purchase-approval", "采购申请由需求人发起，部门负责人审批预算，财务核验付款信息；采购经办人不能同时担任该笔采购的最终验收人。",
         "采购经办人能否担任同一笔采购的最终验收人？", "负责买设备的人可以自己做最终验收吗？",
         "那付款信息由谁核验？"),
    ]),
    ("leave", "星桥科技请假办理指引（合成）", [
        ("annual-leave", "连续三天及以内的年假由直属主管审批；连续超过三天的年假还需部门负责人批准，并登记工作交接人。",
         "连续四天年假需要哪些审批？", "打算连休四天，除了找直属领导还需要做什么？",
         "那只休三天需要谁审批？"),
        ("sick-leave", "病假申请需上传医疗机构出具的诊疗证明；紧急就医可先通知主管，返岗后两个工作日内补齐申请材料。",
         "紧急就医来不及申请病假，返岗后多久补材料？", "突然生病去医院了，手续可以什么时候补办？",
         "那需要上传什么证明？"),
    ]),
    ("security", "星桥科技远程访问与离职权限规则（合成）", [
        ("vpn", "远程访问内部系统必须使用公司VPN并启用多因素认证；个人公共代理不得用于连接公司内部系统。",
         "在家访问内部系统需要使用什么认证？", "远程办公能直接用个人代理连接公司内网吗？",
         "那多因素认证能关闭吗？"),
        ("offboarding", "离职当日18时前由IT管理员停用账号并撤销VPN权限；部门负责人核对共享目录授权，离职人员不得自行导出客户资料。",
         "离职账号和VPN权限由谁在何时停用？", "员工最后一天上班，谁负责关掉远程访问权限？",
         "那共享目录授权由谁核对？"),
    ]),
    ("device", "星桥科技办公设备领用与遗失处理（合成）", [
        ("device-loss", "公司笔记本遗失后须在发现后一小时内通知IT与行政；IT优先冻结设备访问凭证，员工随后提交遗失经过说明。",
         "公司笔记本遗失后应多久通知哪些部门？", "办公电脑找不到了，第一步该联系谁？",
         "那IT接到通知后优先做什么？"),
        ("device-replace", "设备故障先由IT登记维修单；维修预估超过五个工作日可申请备用机，原设备维修完成后两个工作日内归还备用机。",
         "维修多久可以申请备用电脑，何时归还？", "电脑要修一周，能先借一台吗？",
         "那原电脑修好后几天内归还备用机？"),
    ]),
    ("meal", "星桥科技加班餐与客户接待费用规则（合成）", [
        ("overtime-meal", "工作日经批准加班至20时后的餐费每人每次最高35元；不得与同一时段的差旅餐补重复申领。",
         "工作日加班到几点可以报餐费，最高多少？", "晚上八点以后还在加班，吃饭最多能报多少？",
         "那已经领取差旅餐补还能再报吗？"),
        ("client-meal", "客户接待餐费须事前登记接待事由、人数及预算，部门负责人批准后执行；参与接待者本人不能作为唯一审批人。",
         "客户接待餐费事前需要登记哪些信息？", "请客户吃饭之前，需要先报备什么？",
         "那参加接待的人能自己唯一审批吗？"),
    ]),
]


def build():
    documents, evidence, cases = [], [], []
    for key, title, facts in POLICIES:
        # Topic families are held together; security and meal are held-out topics.
        split = "test" if key in {"security", "meal"} else "dev"
        intro = (
            f"{title}\n版本：评测V1；仅用于合成测试。\n"
            "第一条 适用范围。本文件适用于星桥科技内部员工的申请、审批与凭证核验，"
            "承包商和外包人员由合同负责人确认适用范围。正文中的金额与期限均为虚构测试数据，"
            "不得用作真实报销、劳动关系或财务处理依据。办理事项应使用对应业务表单，"
            "不得将采购申请替代出差申请，也不得以聊天中的口头同意替代规定的书面审批。\n"
        )
        parts = [intro]
        for index, (eid, passage, direct, paraphrase, followup) in enumerate(facts, 2):
            parts.append(
                f"第{index}条 办理要求。{passage}\n"
                "申请人应检查姓名、所属部门、发生日期和业务编号是否一致。审批记录及附件由"
                "经办部门按项目编号归档。发现材料不完整时应退回补充，并记录缺失项目和退回原因；"
                "重新提交不得覆盖此前审核意见。工作人员应核对正文适用条件，不得仅凭文件标题判断。\n"
            )
            evidence.append({"id": eid, "document": key, "quote": passage})
            for suffix, question, category, history in (
                ("direct", direct, "direct", []),
                ("paraphrase", paraphrase, "paraphrase", []),
                ("followup", followup, "followup", [direct]),
            ):
                cases.append({"id": f"{eid}-{suffix}", "split": split,
                              "category": category, "question": question,
                              "history": history, "evidence_ids": [eid]})
        parts.append(
            "第四条 查询与复核。申请人对处理结果有异议时可提交原申请编号及具体争议点，"
            "由原经办部门复核。本文件未规定的事项需另行确认，不能类推其他制度的额度或审批权限。"
            "已作废的附件不得作为当前依据；本合成测试仅包含上述版本，不模拟真实企业的发布流程。"
        )
        documents.append({"key": key, "title": title, "version": "synthetic-v1",
                          "content": "\n".join(parts)})
    for cid, question, ids in [
        ("multi-travel", "杭州住宿上限是多少，回程后报销应在几天内提交？", ["hotel", "deadline"]),
        ("multi-purchase", "六千元采购需要几家报价，采购经办人能自己最终验收吗？",
         ["quotes", "purchase-approval"]),
        ("multi-device", "笔记本遗失后多久通知IT，故障维修超过多久可申请备用机？",
         ["device-loss", "device-replace"]),
        ("multi-paperwork", "住宿报销需要哪些凭证，电子发票补不开时能用别的发票替代吗？",
         ["receipt", "lost-invoice"]),
    ]:
        cases.append({"id": cid, "split": "dev", "category": "multi_evidence",
                      "question": question, "evidence_ids": ids})
    for cid, question, split in [
        ("absent-flight", "国际航班商务舱具体可以报销多少元？", "dev"),
        ("absent-marriage", "员工结婚可以休多少天婚假？", "dev"),
        ("absent-password", "公司VPN管理员密码是什么？", "test"),
        ("absent-meal", "周末加班餐费最高报销多少元？", "test"),
    ]:
        cases.append({"id": cid, "split": split, "category": "unanswerable",
                      "question": question, "evidence_ids": [], "answerable": False})
    return {"version": "synthetic-policy-evidence-v1", "synthetic": True,
            "documents": documents, "evidence": evidence, "cases": cases}


if __name__ == "__main__":
    from app.evaluation.retrieval import RetrievalDataset
    dataset = build()
    RetrievalDataset.model_validate(dataset)
    target = ROOT / "examples/retrieval_benchmark_v1.json"
    target.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(dataset['cases'])} cases to {target}")
