from evaluate.EvaluateCore import NormType
from evaluate.NormEngine import NormEngine


def analysis(norms: list, calcNorms: list, extraParams: dict):
    insertedArray = []
    for norm_index, norm in enumerate(norms):
        insertedArray.append([f"{norm['nameEN']}\n{norm['nameZH']}", calcNorms[norm_index], norm['unit'],
                              normToStr(norm, extraParams)])
    return {"SpatiotemporalData": [
        ["参数Parameters", "数值Data", "单位Unit", "参考值Reference"],
        *insertedArray,
        ["检测项共计", "", "", "9 项"]
    ], "SpatiotemporalGraph": []}


def normToStr(norm, extraParams):
    if norm['type'] == NormType.BaseOffsetFloat:
        return f"{norm['rule']['base']} ± {norm['rule']['offset']}"
    elif norm['type'] == NormType.RangeDeepDict:
        normEngine = NormEngine(norm, extraParams)
        targetNorm = normEngine.recursionRule(norm["rule"], norm["ruleHead"])
        return f"≥ {targetNorm}"
