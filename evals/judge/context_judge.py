"""Context-specific judge, using the same configured provider as Waku.

Implementation ships in waku so the installed CLI does not depend on evals.
"""

from waku.context_engineering.eval import ContextJudge

__all__ = ["ContextJudge"]
