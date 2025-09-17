from .base import Agent

# Lightweight package init: avoid importing heavy dependencies (cv2, torch, etc.)
# Provide lazy attribute access for agents to prevent import-time failures in minimal envs.

__all__ = [
    'Agent',
    'AnalyzerHubAgent',
    'MaestroAgent',
    'SegMentorAgent',
    'ScholarSeekerAgent',
    'CrystalForgeAgent',
    'GuardianAgent',
    'MatProphetAgent',
    'ScribeAgent',
]

def __getattr__(name):
    if name == 'AnalyzerHubAgent':
        from .analyzerhub_agent import AnalyzerHubAgent as _T
        return _T
    if name == 'MaestroAgent':
        from .maestro_agent import MaestroAgent as _T
        return _T
    if name == 'SegMentorAgent':
        from .segmentor_agent import SegMentorAgent as _T
        return _T
    if name == 'ScholarSeekerAgent':
        from .scholarseeker_agent import ScholarSeekerAgent as _T
        return _T
    if name == 'CrystalForgeAgent':
        from .crystalforge_agent import CrystalForgeAgent as _T
        return _T
    if name == 'GuardianAgent':
        from .guardian_agent import GuardianAgent as _T
        return _T
    if name == 'MatProphetAgent':
        from .matprophet_agent import MatProphetAgent as _T
        return _T
    if name == 'ScribeAgent':
        from .scribe_agent import ScribeAgent as _T
        return _T
    raise AttributeError(name)
