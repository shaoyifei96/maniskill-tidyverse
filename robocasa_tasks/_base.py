"""
Base imports for RoboCasa task classes ported to ManiSkill.

Replaces `from robocasa.environments.kitchen.kitchen import *`
"""

# Re-export the ManiSkill RoboCasaKitchenEnv with evaluate() override
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv

# Fixture classes used directly in task files (e.g. `Microwave`, `Stove`)
try:
    from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
    from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove
    from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
    from mani_skill.utils.scene_builder.robocasa.fixtures.fridge import Fridge
    from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
    from mani_skill.utils.scene_builder.robocasa.fixtures.dishwasher import Dishwasher
except ImportError:
    pass


class _RandomStateShim:
    """Wraps numpy RandomState to expose Generator-like API (integers, choice, etc.)."""
    def __init__(self, rs):
        self._rs = rs

    def __getattr__(self, name):
        return getattr(self._rs, name)

    def integers(self, low, high=None, size=None, endpoint=False):
        if high is None:
            high, low = low, 0
        if endpoint:
            high = high + 1
        return self._rs.randint(low, high, size=size)

    def choice(self, a, size=None, replace=True, p=None, shuffle=True):
        return self._rs.choice(a, size=size, replace=replace, p=p)

# Fixture types and utilities
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder

# Registration
from mani_skill.utils.registration import register_env

# OU utilities (our SAPIEN port)
from robocasa_tasks import robocasa_utils as OU

# Common imports used by task files
import numpy as np


class Kitchen(RoboCasaKitchenEnv):
    """
    Base class for all ported RoboCasa tasks.
    
    Overrides evaluate() to call _check_success() and return
    {"success": bool} for ManiSkill compatibility.
    """
    
    @property
    def rng(self):
        """Compatibility shim: RoboCasa uses self.rng, ManiSkill uses _batched_episode_rng.
        Returns a numpy Generator (has .integers(), .choice(), etc.)"""
        if hasattr(self, '_batched_episode_rng') and self._batched_episode_rng is not None:
            rng = self._batched_episode_rng[0]
            # _batched_episode_rng may be a numpy Generator already
            if hasattr(rng, 'integers'):
                return rng
            # Wrap RandomState in a Generator-compatible shim
            return _RandomStateShim(rng)
        import numpy as np
        return np.random.default_rng(0)
    
    def evaluate(self):
        if hasattr(self, '_check_success'):
            try:
                success = self._check_success()
                return {"success": bool(success)}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {}
    
    def get_obj_lang(self):
        """Get language description of the main object."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        if "obj" in self.object_actors[scene_idx]:
            return "object"
        return "item"
    
    def check_contact(self, obj1, obj2):
        """Stub for MuJoCo contact check — use proximity instead."""
        # In SAPIEN, we approximate contact with distance
        try:
            scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
            
            # Get positions
            def get_pos(obj):
                if hasattr(obj, 'pose'):
                    return obj.pose.p[0].cpu().numpy()
                if hasattr(obj, 'pos'):
                    return np.array(obj.pos)
                # Try as object name
                if isinstance(obj, str):
                    if obj in self.object_actors[scene_idx]:
                        return self.object_actors[scene_idx][obj]["actor"].pose.p[0].cpu().numpy()
                # Try name attribute
                if hasattr(obj, 'name'):
                    name = obj.name
                    if name in self.object_actors[scene_idx]:
                        return self.object_actors[scene_idx][name]["actor"].pose.p[0].cpu().numpy()
                return None
            
            p1 = get_pos(obj1)
            p2 = get_pos(obj2)
            if p1 is not None and p2 is not None:
                return float(np.linalg.norm(p1 - p2)) < 0.15
        except Exception:
            pass
        return False
    
    def get_fixture(self, fixture_id=None, ref=None, ref_fixture=None, id=None, size=None):
        """Get a fixture by id (compatibility with RoboCasa API).
        
        Args:
            fixture_id: FixtureType or string id (positional)
            id: same as fixture_id (keyword form used by some tasks)
            ref: optional reference fixture for proximity search
            ref_fixture: alias for ref
            size: minimum size for counter fixtures
        """
        if id is not None and fixture_id is None:
            fixture_id = id
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        # Delegate to scene builder if possible
        if hasattr(self, 'scene_builder') and self.scene_builder is not None:
            fixtures = self.scene_builder.scene_data[scene_idx]["fixtures"]
            try:
                return self.scene_builder.get_fixture(fixtures, fixture_id,
                                                       ref=ref or ref_fixture)
            except Exception:
                pass
        # Fallback: search fixture_refs
        if isinstance(fixture_id, str) and fixture_id in self.fixture_refs[scene_idx]:
            return self.fixture_refs[scene_idx][fixture_id]
        return None

    @property
    def sim(self):
        """Stub: RoboCasa tasks call self.sim (MuJoCo). Return None-safe shim."""
        return _SimShim()

    def _get_objects_dict(self):
        """RoboCasa compat: returns object dict for current env index."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        return {k: v["actor"] for k, v in self.object_actors[scene_idx].items()}

    def _get_fixtures_dict(self):
        """RoboCasa compat: returns fixture dict for current env index."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        return self.fixture_refs[scene_idx]


class _SimShim:
    """Null-safe shim for self.sim calls from RoboCasa _check_success methods."""
    class _Data:
        def __getattr__(self, name):
            return None
    data = _Data()
    def __getattr__(self, name):
        return None
