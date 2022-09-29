from evaluate import PluginGaitAnalysis, PluginSLBAnalysis
from evaluate.EvaluateCore import AnalysisReport


class ReportAnalysisFactory:
    def __init__(self, analysisReport: AnalysisReport):
        self.analysisReport = analysisReport

    def exec(self, *args, **kwargs):
        analysisResult = {}
        if self.analysisReport == AnalysisReport.Gait:
            analysisResult = PluginGaitAnalysis.analysis(df_angles=kwargs["df_angles"], pts_cam=kwargs["pts_cams"],
                                                         analysis_keypoint=kwargs["landmark"])
        elif self.analysisReport == AnalysisReport.SLB:
            analysisResult = PluginSLBAnalysis.analysis(norms=kwargs["norms"], calcNorms=kwargs["calcNorms"],
                                                        extraParams=kwargs["extraParams"])

        return analysisResult
