from enum import Enum


class RequireCollect(Enum):
    age = 1
    gender = 2
    side = 3
    eyesClosed = 4
    name = 5


class PartAngle(Enum):
    Knee = 1
    Hip = 2
    Pelvis = 3
    Ankle = 4


class NormType(Enum):
    RangeDeepDict = 1
    BaseOffsetFloat = 2


PART_ANGLE_ALL = {
    PartAngle.Knee: {
        "name": "膝关节",
        "item": ["LKnee_angle", "RKnee_angle"]
    },
    PartAngle.Hip: {
        "name": "髋关节",
        "item": ["TorsoLFemur_angle", "TorsoRFemur_angle", "LHip_angle", "RHip_angle", "LTibiaSelf_vector",
                 "RTibiaSelf_vector"]
    },
    PartAngle.Pelvis: {
        "name": "骨盆",
        "item": ["TorsoLHip_angle", "TorsoRHip_angle"]
    },
    PartAngle.Ankle: {
        "name": "踝关节",
        "item": ["LAnkle_angle", "RAnkle_angle"]
    }
}

InfoForm = {
    RequireCollect.age: {
        "type": "slider",
        "title": "评测人员年龄{}",
        "range": [1, 100],
        "defaultValue": 25,
        "step": 1
    },
    RequireCollect.gender: {
        "type": "select",
        "title": "评测人员性别",
        "item": ["男", "女"],
        "_item": ["Male", "Female"]
    },
    RequireCollect.side: {
        "type": "select",
        "title": "评估动作左右侧",
        "item": ["左", "右"],
        "_item": ["left", "right"]
    },
    RequireCollect.eyesClosed: {
        "type": "select",
        "title": "评估是否闭眼",
        "item": ["是", "否"],
        "_item": [True, False]
    },
    RequireCollect.name: {
        "type": "input",
        "title": "受测者姓名"
    }
}

EvaluateMetadata = [
    {
        "name": "步态分析",
        "requireCollect": [
            RequireCollect.name,
            RequireCollect.age,
            RequireCollect.gender
        ],
        "part": [PartAngle.Knee, PartAngle.Hip, PartAngle.Pelvis, PartAngle.Ankle],
        "calcRules": "",
        "result": {},
        "norms": {}
    },
    {
        "name": "单腿桥SLB",
        "requireCollect": [
            RequireCollect.name,
            RequireCollect.side
        ],
        "part": [PartAngle.Knee, PartAngle.Hip, PartAngle.Pelvis, PartAngle.Ankle],
        "calcRules": "",
        "result": {
            "nameEN": "StickTime",
            "nameZH": "坚持时间",
            "unit": "sec"
        },
        "norms": {
            "type": NormType.BaseOffsetFloat,
            "unit": "sec",
            "rule": {
                "base": 23,
                "offset": 16.5
            }
        }
    },
    {
        "name": "单腿站SLS",
        "part": [PartAngle.Knee, PartAngle.Hip, PartAngle.Pelvis, PartAngle.Ankle],
        "requireCollect": [
            RequireCollect.name,
            RequireCollect.age,
            RequireCollect.gender,
            RequireCollect.eyesClosed
        ],
        # 躯干与地面的夹角在(0, 50)范围内，并且股骨与地面的夹角大于30°，并且胫骨与地面的夹角大于30°,
        "calcRules": "(angle(ly{$torso},{$torso}) in range(0, 50)) && angle(ly({$femur}),{$femur})>30 && angle(ly({$tibia}),{$tibia})>30",
        "result": {
            "nameEN": "StickTime",
            "nameZH": "坚持时间",
            "unit": "sec"
        },
        "norms": {
            "type": NormType.RangeDeepDict,
            "unit": "sec",
            "rule": {
                "18-39": {
                    "Female": {
                        "EyesOpen": 43.5,
                        "EyesClosed": 8.5,
                    },
                    "Male": {
                        "EyesOpen": 43.2,
                        "EyesClosed": 10.2,
                    }
                },
                "40-49": {
                    "Female": {
                        "EyesOpen": 40.4,
                        "EyesClosed": 7.4,
                    },
                    "Male": {
                        "EyesOpen": 40.1,
                        "EyesClosed": 7.3,
                    }
                },
                "50-59": {
                    "Female": {
                        "EyesOpen": 36,
                        "EyesClosed": 5.0,
                    },
                    "Male": {
                        "EyesOpen": 38.1,
                        "EyesClosed": 4.5,
                    }
                },
                "60, 69": {
                    "Female": {
                        "EyesOpen": 25.1,
                        "EyesClosed": 2.5,
                    },
                    "Male": {
                        "EyesOpen": 28.7,
                        "EyesClosed": 3.1,
                    }
                },
                "70, 79": {
                    "Female": {
                        "EyesOpen": 11.3,
                        "EyesClosed": 2.2,
                    },
                    "Male": {
                        "EyesOpen": 18.3,
                        "EyesClosed": 1.9,
                    }
                },
                "80, 99": {
                    "Female": {
                        "EyesOpen": 7.4,
                        "EyesClosed": 1.4,
                    },
                    "Male": {
                        "EyesOpen": 5.6,
                        "EyesClosed": 1.3,
                    }
                }
            }
        }
    }
]
