from enum import Enum


class RequireCollect(Enum):
    age = 1
    gender = 2
    side = 3
    eyesClosed = 4
    name = 5
    time = 6


class PartAngle(Enum):
    Knee = 1
    Hip = 2
    Pelvis = 3
    Ankle = 4


class NormType(Enum):
    RangeDeepDict = 1
    BaseOffsetFloat = 2
    NoneType = 3


class AnalysisReport(Enum):
    Gait = 1
    SLB = 2
    SLS = 3


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
    },
    RequireCollect.time: {
        "type": "spinbox",
        "title": "检测时长/sec",
        "defaultValue": 20,
    }
}

EvaluateMetadata = [
    {
        "name": "步态分析Gait",
        "requireCollect": [
            RequireCollect.name,
            RequireCollect.age,
            RequireCollect.gender,
            RequireCollect.time
        ],
        "part": [PartAngle.Knee, PartAngle.Hip, PartAngle.Pelvis, PartAngle.Ankle],
        "venv": ["$time", "$keypoints"],
        "calcRules": {
            "credit": [i for i in range(33)],
            "start": "credible_pose({$keypoints})>0.5",
            "interrupt": "credible_pose({$keypoints})<=0.5",
            "end": "(currentTime() - {$detectStartTime}) > {$time}"
        },
        "patientTips": {
            "onBeforeDetect": "调整姿势使身体和四肢完全包含在相机视图中",
            "onFirstDetect": "已识别到所有检测点，开始检测，请开始行走",
            "onDetecting": "'已检测' + str(round(currentTime() - {$detectStartTime}, 1)) + '秒，剩余' + str(round({$time} - (currentTime() - {$detectStartTime}),1)) + '秒'",
            "onDetectingInterrupt": "检测过程被打断！重新调整姿势，并等待重新检测",
            "onDetectEnd": "检测完成，等待生成报告"
        },
        "sequenceLog": {
            "onBeforeDetect": "调整姿势使身体和四肢完全包含在相机视图中",
            "onFirstDetect": "已识别到所有检测点，开始检测",
            "onDetecting": "'已检测' + str(currentTime() - {$detectStartTime}) + '秒'",
            "onDetectingInterrupt": "检测过程被打断！等待重新检测",
            "onDetectEnd": "检测结束"
        },
        "EchoNumber": "str(({$k23[2]}+{$k24[2]}+{$k11[2]}+{$k12[2]})/4)+'m'",
        "output": {
            "analysisReport": AnalysisReport.Gait,
            "general": []
        },
    },
    {
        "name": "单腿桥SLB",
        "requireCollect": [
            RequireCollect.name,
            RequireCollect.side
        ],
        "part": [PartAngle.Knee, PartAngle.Hip, PartAngle.Pelvis, PartAngle.Ankle],
        "venv": ["$torso", "$L$femur", "$R$femur", "$L$tibia", "$R$tibia"],
        # 躯干与地面的夹角在(0, 50)范围内，并且股骨与地面的夹角大于30°，并且胫骨与地面的夹角大于30°,
        "calcRules": {
            "credit": [11, 12, 23, 24, 25, 26, 27, 28],
            "start": "(angle(ly(lz({$torso})),lz({$torso}), m=True) <= 45) and angle(ly(lz({$femur})),lz({$femur}), m=True)>30 and angle(reverse(lz({$femur})),reverse(lz({$tibia})))<41",
            "interrupt": "False",
            "end": "not ((angle(ly(lz({$torso})),lz({$torso}), m=True) <= 45) and angle(ly(lz({$femur})),lz({$femur}), m=True)>30 and angle(reverse(lz({$femur})),reverse(lz({$tibia})))<41)"
        },
        "patientTips": {
            "onBeforeDetect": "开始仰卧，双膝弯曲，双脚放在地板上，伸直一\n条腿，使其与另一条腿保持一条直线，然后收紧腹部，\n将臀部抬离地板，形成桥式姿势。",
            "onFirstDetect": "已达到动作要求，开始计时，请坚持",
            "onDetecting": "'已坚持' + str(round(currentTime() - {$detectStartTime}, 1)) + '秒'",
            "onDetectingInterrupt": "",
            "onDetectEnd": "检测完成，等待生成报告"
        },
        "sequenceLog": {
            "onBeforeDetect": "开始仰卧，双膝弯曲，双脚放在地板上，伸直一条腿，使其与另一条腿保持一条直线，然后收紧腹部，将臀部抬离地板，形成桥式姿势。",
            "onFirstDetect": "已达到动作要求，开始检测",
            "onDetecting": "'已坚持' + str(currentTime() - {$detectStartTime}) + '秒'",
            "onDetectingInterrupt": "",
            "onDetectEnd": "检测结束"
        },
        "EchoNumber": "str(round(currentTime() - {$detectStartTime}, 1))+'s'",
        "output": {
            "analysisReport": AnalysisReport.SLB,
            "general": [
                {
                    "nameEN": "StickTime",
                    "nameZH": "坚持时间",
                    "unit": "sec",
                    "calcRule": "round(currentTime() - {$detectStartTime}, 1)",
                    "norm": {
                        "type": NormType.BaseOffsetFloat,
                        "unit": "sec",
                        "rule": {
                            "base": 23.0,
                            "offset": 16.5
                        }
                    }
                }
            ]
        },
    },
    {
        "name": "单腿站SLS",
        "requireCollect": [
            RequireCollect.name,
            RequireCollect.age,
            RequireCollect.gender,
            RequireCollect.eyesClosed,
            RequireCollect.side
        ],
        "part": [PartAngle.Knee, PartAngle.Hip, PartAngle.Pelvis, PartAngle.Ankle],
        "venv": ["$torso", "$L$femur", "$R$femur", "$L$tibia", "$R$tibia"],
        "calcRules": {
            "credit": [11, 12, 23, 24, 25, 26, 27, 28],
            "start": "(70 <= angle(ly({$torso}),{$torso}, m=True) <= 120)",
            "interrupt": "False",
            "end": "not (70 <= angle(ly({$torso}),{$torso}, m=True) <= 120)"
        },
        "patientTips": {
            "onBeforeDetect": "以直立姿势开始，双脚并拢，双臂放在身体两侧。\n将一只脚抬离地板，用另一条腿保持平衡。 ",
            "onFirstDetect": "已达到动作要求，开始计时，请坚持",
            "onDetecting": "'已坚持' + str(round(currentTime() - {$detectStartTime}, 1)) + '秒'",
            "onDetectingInterrupt": "",
            "onDetectEnd": "检测完成，等待生成报告"
        },
        "sequenceLog": {
            "onBeforeDetect": "以直立姿势开始，双脚并拢，双臂放在身体两侧。将一只脚抬离地板，用另一条腿保持平衡。 在这个位置保持平衡。尽量不要将手臂从身体上移开或让体重从一侧转移到另一侧。",
            "onFirstDetect": "已达到动作要求，开始检测",
            "onDetecting": "'已坚持' + str(currentTime() - {$detectStartTime}) + '秒'",
            "onDetectingInterrupt": "",
            "onDetectEnd": "检测结束"
        },
        "EchoNumber": "str(round(currentTime() - {$detectStartTime}, 1))+'s'",
        "output": {
            "analysisReport": AnalysisReport.SLS,
            "general": [{
                "nameEN": "StickTime",
                "nameZH": "坚持时间",
                "unit": "sec",
                "calcRule": "round(currentTime() - {$detectStartTime}, 1)",
                "norm": {
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
                }
            }]
        }
    }
]
