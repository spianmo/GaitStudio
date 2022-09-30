from evaluate.EvaluateCore import NormType, EvaluateMetadata


class NormEngine:
    def __init__(self, norms, extraParams):
        self.norms = norms
        self.extraParams = {}
        for infoItem in extraParams.items():
            self.extraParams[f"{infoItem[0].name}"] = infoItem[1]

    def exec(self, src):
        if self.norms["type"] == NormType.BaseOffsetFloat:
            return {
                "result": self.norms["rule"]["base"] - self.norms["rule"]["offset"] <= float(src) <= self.norms["rule"][
                    "base"] + self.norms["rule"]["offset"],
                "resultDetail": src,
                **self.norms
            }
        elif self.norms["type"] == NormType.RangeDeepDict:
            targetNorm = self.recursionRule(self.norms["rule"], self.norms["ruleHead"])
            return {
                "result": src >= targetNorm,
                "resultDetail": src,
                **self.norms
            }

    def recursionRule(self, rules, ruleHead):
        index = 0
        if type(list(rules.keys())[0]) is tuple:
            firstRuleLevel = rules[self.detectTargetRangle(rules, self.extraParams[ruleHead[index]])]
        else:
            firstRuleLevel = rules[self.extraParams[ruleHead[0]]]
        return self._recursionRule(firstRuleLevel, ruleHead, index)

    def _recursionRule(self, rules, ruleHead, index):
        index = index + 1
        if type(rules) is not dict:
            return rules
        if type(list(rules.keys())[0]) is tuple:
            currentLevel = rules[self.detectTargetRangle(rules, self.extraParams[ruleHead[index]])]
        else:
            currentLevel = rules[self.extraParams[ruleHead[index]]]
        return self._recursionRule(currentLevel, ruleHead, index)

    def detectTargetRangle(self, rules, currentPropValue):
        keys = list(rules.keys())
        for tupleKey in keys:
            if currentPropValue in range(tupleKey[0], tupleKey[1]):
                return tupleKey


if __name__ == '__main__':
    normEngine = NormEngine({
        "type": NormType.RangeDeepDict,
        "unit": "sec",
        "ruleHead": ["age", "gender", "eyesClosed"],
        "rule": {
            (18, 39): {
                "Female": {
                    False: 43.5,
                    True: 8.5,
                },
                "Male": {
                    False: 43.2,
                    True: 10.2,
                }
            },
            (40, 49): {
                "Female": {
                    False: 40.4,
                    True: 7.4,
                },
                "Male": {
                    False: 40.1,
                    True: 7.3,
                }
            },
            (50, 59): {
                "Female": {
                    False: 36,
                    True: 5.0,
                },
                "Male": {
                    False: 38.1,
                    True: 4.5,
                }
            },
            (60, 69): {
                "Female": {
                    False: 25.1,
                    True: 2.5,
                },
                "Male": {
                    False: 28.7,
                    True: 3.1,
                }
            },
            (70, 79): {
                "Female": {
                    False: 11.3,
                    True: 2.2,
                },
                "Male": {
                    False: 18.3,
                    True: 1.9,
                }
            },
            (80, 99): {
                "Female": {
                    False: 7.4,
                    True: 1.4,
                },
                "Male": {
                    False: 5.6,
                    True: 1.3,
                }
            }
        }
    }, {"age": 24, "gender": "Male", "eyesClosed": True})
    print(normEngine.exec(50))
