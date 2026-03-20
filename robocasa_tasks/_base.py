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
        # RandomState.choice only supports 1-D arrays or int
        # Generator.choice supports multi-dimensional. Emulate it:
        if isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], (list, tuple)):
            idx = self._rs.randint(0, len(a))
            return a[idx]
        try:
            return self._rs.choice(a, size=size, replace=replace, p=p)
        except ValueError:
            # Fallback for empty or weird input
            if len(a) == 0:
                raise
            idx = self._rs.randint(0, len(a))
            return a[idx]

# Fixture types and utilities
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder

# Registration
from mani_skill.utils.registration import register_env

# OU utilities (our SAPIEN port)
from robocasa_tasks import robocasa_utils as OU

# Common imports used by task files
import numpy as np


class _FixtureRefsProxy:
    """
    Proxy that makes self.fixture_refs behave like a dict (RoboCasa API)
    while the underlying ManiSkill storage is a list-of-dicts.
    """
    def __init__(self, refs_list, idx):
        object.__setattr__(self, '_refs_list', refs_list)
        object.__setattr__(self, '_idx', idx)

    def _ensure_idx(self, idx):
        refs_list = object.__getattribute__(self, '_refs_list')
        while len(refs_list) <= idx:
            refs_list.append({})

    def __getitem__(self, key):
        if isinstance(key, int):
            refs_list = object.__getattribute__(self, '_refs_list')
            return refs_list[key]
        idx = object.__getattribute__(self, '_idx')
        self._ensure_idx(idx)
        refs_list = object.__getattribute__(self, '_refs_list')
        return refs_list[idx][key]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            refs_list = object.__getattribute__(self, '_refs_list')
            refs_list[key] = value
            return
        idx = object.__getattribute__(self, '_idx')
        self._ensure_idx(idx)
        refs_list = object.__getattribute__(self, '_refs_list')
        refs_list[idx][key] = value

    def __contains__(self, key):
        if isinstance(key, int):
            refs_list = object.__getattribute__(self, '_refs_list')
            return key < len(refs_list)
        idx = object.__getattribute__(self, '_idx')
        self._ensure_idx(idx)
        refs_list = object.__getattribute__(self, '_refs_list')
        return key in refs_list[idx]

    def __iter__(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return iter(refs_list[idx])
        return iter({})

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def items(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return refs_list[idx].items()
        return {}.items()

    def keys(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return refs_list[idx].keys()
        return {}.keys()

    def values(self):
        idx = object.__getattribute__(self, '_idx')
        refs_list = object.__getattribute__(self, '_refs_list')
        if idx < len(refs_list):
            return refs_list[idx].values()
        return {}.values()

    def append(self, item):
        """Support list-style append (ManiSkill uses fixture_refs.append({}))."""
        refs_list = object.__getattribute__(self, '_refs_list')
        refs_list.append(item)
        # Update idx to point to newly appended item
        object.__setattr__(self, '_idx', len(refs_list) - 1)

    def __len__(self):
        refs_list = object.__getattribute__(self, '_refs_list')
        return len(refs_list)


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
            # Lazy-init attributes that _reset_internal would have set
            self._ensure_reset_attrs()
            try:
                success = self._check_success()
                return {"success": bool(success)}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {}

    def compute_robot_base_placement_pose(self, fixture, **kwargs):
        """Stub for RoboCasa's base placement computation. Returns fixture pos + facing direction."""
        pos = np.array(getattr(fixture, 'pos', [0, 0, 0]))[:2]
        # Default: stand 0.5m in front of fixture (facing it)
        offset = np.array([0.5, 0.0])
        return pos + offset, 0.0

    def _ensure_reset_attrs(self):
        """Set attributes that _reset_internal() normally sets, using SAPIEN state."""
        # self.knob — used by stove tasks (SearingMeat, MultistepSteaming, etc.)
        if not hasattr(self, 'knob') and hasattr(self, 'stove') and hasattr(self, 'knob_id'):
            try:
                valid = list(self.stove.get_knobs_state(env=self).keys())
                if valid:
                    self.knob = valid[0]  # default to first knob
            except Exception:
                pass
        # self.target_location — used by some navigate tasks
        if not hasattr(self, 'target_location') and hasattr(self, 'init_robot_base_pos'):
            self.target_location = getattr(self, 'init_robot_base_pos', None)
    
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
    
    def __getattribute__(self, name):
        if name in ('fixture_refs', 'objects'):
            refs_list = object.__getattribute__(self, name)
            if isinstance(refs_list, _FixtureRefsProxy):
                return refs_list
            if isinstance(refs_list, list):
                idx = object.__getattribute__(self, '_scene_idx_to_be_loaded') if hasattr(self, '_scene_idx_to_be_loaded') else 0
                return _FixtureRefsProxy(refs_list, idx)
            return refs_list
        return object.__getattribute__(self, name)

    @property
    def fixtures(self):
        """RoboCasa-compatible access to scene fixtures dict."""
        idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        if hasattr(self, 'scene_builder') and hasattr(self.scene_builder, 'scene_data'):
            if idx < len(self.scene_builder.scene_data):
                return self.scene_builder.scene_data[idx].get("fixtures", {})
        return {}

    def _load_scene(self, options):
        """Ensure _setup_kitchen_references is always called, even for tasks without _get_obj_cfgs."""
        # Check if parent would skip _setup_kitchen_references
        if not self.fixtures_only and not hasattr(self, '_get_obj_cfgs'):
            # Provide a minimal _get_obj_cfgs so parent runs _setup_kitchen_references
            self._get_obj_cfgs = lambda: []
        result = super()._load_scene(options)
        # Try to run _reset_internal to set task-specific attrs (e.g. self.knob, self.target)
        # Guard: only if it's defined on the subclass (not the RoboCasa base)
        if hasattr(type(self), '_reset_internal') and type(self)._reset_internal is not \
                getattr(type(self).__mro__[1], '_reset_internal', None):
            try:
                type(self)._reset_internal(self)
            except Exception:
                pass  # Silently ignore SAPIEN-incompatible parts
        return result

    def register_fixture_ref(self, ref_name, fn_kwargs):
        """Override to handle missing fixture types gracefully."""
        try:
            return super().register_fixture_ref(ref_name, fn_kwargs)
        except (KeyError, AssertionError):
            # Fixture type not present in this scene layout — try fallback to COUNTER
            scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
            raw_refs = object.__getattribute__(self, 'fixture_refs')
            try:
                from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
                fallback = self.scene_builder.get_fixture(
                    self.scene_builder.scene_data[scene_idx]["fixtures"],
                    id=FixtureType.COUNTER
                )
                raw_refs[scene_idx][ref_name] = fallback
                return fallback
            except Exception:
                raw_refs[scene_idx][ref_name] = None
                return None

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
        """Stub: RoboCasa tasks call self.sim (MuJoCo). Return env-aware shim."""
        return _SimShim(self)

    @property
    def obj_body_id(self):
        """RoboCasa compat: maps object name → integer index (used for body_xpos lookup)."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        mapping = {}
        for i, (name, info) in enumerate(self.object_actors[scene_idx].items()):
            mapping[name] = i
            # Also map via the object's .name attribute if different
            actor = info.get("actor")
            if actor and hasattr(actor, 'name') and actor.name != name:
                mapping[actor.name] = i
        return mapping

    def _get_objects_dict(self):
        """RoboCasa compat: returns object dict for current env index."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        return {k: v["actor"] for k, v in self.object_actors[scene_idx].items()}

    def _get_fixtures_dict(self):
        """RoboCasa compat: returns fixture dict for current env index."""
        scene_idx = getattr(self, '_scene_idx_to_be_loaded', 0)
        return self.fixture_refs[scene_idx]


class _SimShim:
    """Env-aware shim for self.sim calls from RoboCasa _check_success methods."""
    model = None  # No MuJoCo model

    def __init__(self, env=None):
        self._env = env
        self.data = _SimData(env)

    def __getattr__(self, name):
        return None


class _SimData:
    """Shim for sim.data — provides body_xpos and get_site_xpos via SAPIEN."""

    def __init__(self, env):
        self._env = env

    @property
    def body_xpos(self):
        """Returns a dict-like object: body_xpos[idx] → position array."""
        return _BodyXposProxy(self._env)

    def get_site_xpos(self, site_name):
        """Look up site position from articulation XML data."""
        env = self._env
        if env is None:
            return np.zeros(3)
        # Search all articulations for a site matching this name
        for art in env.scene.get_all_articulations():
            loader = getattr(art, 'loader', None)
            if loader is None:
                continue
            xml = getattr(loader, 'xml', None)
            if xml is None:
                continue
            site = xml.find(f".//*site[@name='{site_name}']")
            if site is not None:
                pos_str = site.get('pos', '0 0 0')
                local_pos = np.array([float(x) for x in pos_str.split()])
                # Transform by articulation root pose
                art_pos = art.pose.p
                if hasattr(art_pos, 'cpu'):
                    art_pos = art_pos[0].cpu().numpy()
                return np.array(art_pos) + local_pos
        # Fallback: search fixtures
        scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
        try:
            for fx_name, fx in env.fixture_refs[scene_idx].items():
                if hasattr(fx, 'loader') and fx.loader is not None:
                    xml = getattr(fx.loader, 'xml', None)
                    if xml is None:
                        continue
                    site = xml.find(f".//*site[@name='{site_name}']")
                    if site is not None:
                        pos_str = site.get('pos', '0 0 0')
                        local_pos = np.array([float(x) for x in pos_str.split()])
                        fx_pos = np.array(getattr(fx, 'pos', [0, 0, 0]))
                        return fx_pos + local_pos
        except Exception:
            pass
        return np.zeros(3)

    @property
    def qpos(self):
        """Allow sim.data.qpos[joint_id] — return zeros (joints not tracked this way)."""
        return _QposProxy(self._env)

    def set_joint_qpos(self, *args, **kwargs):
        pass  # no-op in eval mode

    @property
    def xquat(self):
        """sim.data.xquat[idx] → identity quaternion proxy."""
        return _QuatProxy(self._env)

    @property
    def site_xpos(self):
        """sim.data.site_xpos[site_id] → zeros proxy (site ids not tracked)."""
        return _ZerosProxy()

    def __getattr__(self, name):
        return _ZerosProxy()  # Safe fallback: returns zeros for any attribute access


class _ZerosProxy:
    """Safe fallback proxy that returns np.zeros(3) for any index."""
    def __getitem__(self, idx):
        return np.zeros(3)
    def __call__(self, *a, **kw):
        return np.zeros(3)


class _QuatProxy:
    """sim.data.xquat[idx] → identity quaternion [1,0,0,0]."""
    def __init__(self, env):
        self._env = env

    def __getitem__(self, idx):
        import numpy as _np
        # Try to get actual orientation from actor
        env = self._env
        if env is not None:
            scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
            actors = list(env.object_actors[scene_idx].values())
            if isinstance(idx, int) and 0 <= idx < len(actors):
                actor = actors[idx]["actor"]
                q = actor.pose.q
                if hasattr(q, 'cpu'):
                    return q[0].cpu().numpy()
                return _np.array(q)
        return _np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion


class _BodyXposProxy:
    """body_xpos[idx] → object position from SAPIEN actors."""
    def __init__(self, env):
        self._env = env

    def __getitem__(self, idx):
        env = self._env
        if env is None:
            return np.zeros(3)
        scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
        actors = list(env.object_actors[scene_idx].values())
        if isinstance(idx, int) and 0 <= idx < len(actors):
            actor = actors[idx]["actor"]
            p = actor.pose.p
            if hasattr(p, 'cpu'):
                return p[0].cpu().numpy()
            return np.array(p)
        return np.zeros(3)


class _QposProxy:
    """sim.data.qpos[joint_id] → 0.0 fallback."""
    def __init__(self, env):
        self._env = env

    def __getitem__(self, idx):
        return 0.0
