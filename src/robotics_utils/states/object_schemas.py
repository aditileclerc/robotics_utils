from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from robotics_utils.spatial import DEFAULT_FRAME, Pose3D

if TYPE_CHECKING:
    # Imported lazily: robotics_utils.kinematics pulls in the vision stack (torch, tk),
    # and Configuration is only a type alias for Dict[str, float].
    from robotics_utils.kinematics.configuration import Configuration


class _Unobserved:
    """Sentinel marking a field that perception did not report."""

    _instance: Optional[_Unobserved] = None

    def __new__(cls) -> _Unobserved:
        """Return the single shared instance, so equality and identity agree."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return a readable representation."""
        return "UNOBSERVED"

    def __bool__(self) -> bool:
        """Treat an unobserved field as falsy."""
        return False


UNOBSERVED = _Unobserved()
"""Marks a field perception never reported, as distinct from an observed absence.

`dirty=False` means the board was looked at and is clean. `dirty=UNOBSERVED` means nobody
determined it. Collapsing both into None makes those indistinguishable, which mislabels
the transitions used to learn a skill's initiation set.

In YAML the distinction is carried by presence: an unobserved field is left out of the
file, an observed one is written explicitly.
"""


def pose_to_yaml(pose: Union[Pose3D, None, _Unobserved]) -> Any:
    """Convert a pose field to plain YAML data (floats, never numpy scalars)."""
    if not isinstance(pose, Pose3D):
        return None
    return {"xyz_rpy": [float(v) for v in pose.to_xyz_rpy()], "frame": pose.ref_frame}


def pose_from_yaml(data: Any) -> Union[Pose3D, None, _Unobserved]:
    """Construct a pose from YAML data, or return None if it was an observed absence."""
    if data is None:
        return None
    return Pose3D.from_yaml_data(data, DEFAULT_FRAME)


@dataclass(frozen=True)
class RobotState:
    """Store the robot pose, arm configuration, and gripper position."""

    pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED
    arm: Union[Configuration, None, _Unobserved] = UNOBSERVED
    gripper: Union[float, None, _Unobserved] = UNOBSERVED
    """Gripper joint angle in RADIANS: about 0.0 closed, -1.57 fully open on Spot."""

    def to_yaml_data(self) -> Dict[str, Any]:
        """Convert the robot state into YAML data, omitting unobserved fields."""
        data: Dict[str, Any] = {}
        if self.pose is not UNOBSERVED:
            data["pose"] = pose_to_yaml(self.pose)
        if self.arm is not UNOBSERVED:
            data["arm"] = self.arm
        if self.gripper is not UNOBSERVED:
            data["gripper"] = self.gripper
        return data

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> RobotState:
        """Construct a robot state from YAML data."""
        return cls(
            pose=pose_from_yaml(data["pose"]) if "pose" in data else UNOBSERVED,
            arm=data.get("arm", UNOBSERVED),
            gripper=data.get("gripper", UNOBSERVED),
        )


@dataclass(frozen=True)
class EraserState:
    """Store the pose of the eraser and which robot is holding it."""

    pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED
    held_by: Union[str, None, _Unobserved] = UNOBSERVED
    """Name of the robot holding the eraser, None if free-standing, UNOBSERVED if unknown."""

    def to_yaml_data(self) -> Dict[str, Any]:
        """Convert the eraser state into YAML data, omitting unobserved fields."""
        data: Dict[str, Any] = {"type": "eraser"}
        if self.pose is not UNOBSERVED:
            data["pose"] = pose_to_yaml(self.pose)
        if self.held_by is not UNOBSERVED:
            data["held_by"] = self.held_by
        return data

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> EraserState:
        """Construct an eraser state from YAML data."""
        return cls(
            pose=pose_from_yaml(data["pose"]) if "pose" in data else UNOBSERVED,
            held_by=data.get("held_by", UNOBSERVED),
        )


@dataclass(frozen=True)
class BoardState:
    """Store the pose of a writable board and whether it has writing on it."""

    pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED
    dirty: Union[bool, None, _Unobserved] = UNOBSERVED
    """Whether the board has writing on it. Erase(?board) depends on this."""

    def to_yaml_data(self) -> Dict[str, Any]:
        """Convert the board state into YAML data, omitting unobserved fields."""
        data: Dict[str, Any] = {"type": "board"}
        if self.pose is not UNOBSERVED:
            data["pose"] = pose_to_yaml(self.pose)
        if self.dirty is not UNOBSERVED:
            data["dirty"] = self.dirty
        return data

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> BoardState:
        """Construct a board state from YAML data."""
        return cls(
            pose=pose_from_yaml(data["pose"]) if "pose" in data else UNOBSERVED,
            dirty=data.get("dirty", UNOBSERVED),
        )


@dataclass(frozen=True)
class DrawerState:
    """Store the pose and articulation state of a drawer."""

    pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED
    """Pose of the cabinet body, which does not move when the drawer opens."""

    open: Union[bool, None, _Unobserved] = UNOBSERVED
    face_pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED
    """Pose of the drawer face, which does move. An open drawer is an obstruction, and a
    boolean alone cannot express where it sticks out to."""

    def to_yaml_data(self) -> Dict[str, Any]:
        """Convert the drawer state into YAML data, omitting unobserved fields."""
        data: Dict[str, Any] = {"type": "drawer"}
        if self.pose is not UNOBSERVED:
            data["pose"] = pose_to_yaml(self.pose)
        if self.open is not UNOBSERVED:
            data["open"] = self.open
        if self.face_pose is not UNOBSERVED:
            data["face_pose"] = pose_to_yaml(self.face_pose)
        return data

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> DrawerState:
        """Construct a drawer state from YAML data."""
        return cls(
            pose=pose_from_yaml(data["pose"]) if "pose" in data else UNOBSERVED,
            open=data.get("open", UNOBSERVED),
            face_pose=pose_from_yaml(data["face_pose"]) if "face_pose" in data else UNOBSERVED,
        )


@dataclass(frozen=True)
class DoorState:
    """Store the pose and articulation state of a door."""

    pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED
    open: Union[bool, None, _Unobserved] = UNOBSERVED

    def to_yaml_data(self) -> Dict[str, Any]:
        """Convert the door state into YAML data, omitting unobserved fields."""
        data: Dict[str, Any] = {"type": "door"}
        if self.pose is not UNOBSERVED:
            data["pose"] = pose_to_yaml(self.pose)
        if self.open is not UNOBSERVED:
            data["open"] = self.open
        return data

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> DoorState:
        """Construct a door state from YAML data."""
        return cls(
            pose=pose_from_yaml(data["pose"]) if "pose" in data else UNOBSERVED,
            open=data.get("open", UNOBSERVED),
        )


@dataclass(frozen=True)
class SurfaceState:
    """Store the pose of a surface that objects can be placed onto."""

    pose: Union[Pose3D, None, _Unobserved] = UNOBSERVED

    def to_yaml_data(self) -> Dict[str, Any]:
        """Convert the surface state into YAML data, omitting unobserved fields."""
        data: Dict[str, Any] = {"type": "surface"}
        if self.pose is not UNOBSERVED:
            data["pose"] = pose_to_yaml(self.pose)
        return data

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> SurfaceState:
        """Construct a surface state from YAML data."""
        return cls(pose=pose_from_yaml(data["pose"]) if "pose" in data else UNOBSERVED)


OBJECT_STATE_TYPES = {
    "eraser": EraserState,
    "board": BoardState,
    "drawer": DrawerState,
    "door": DoorState,
    "surface": SurfaceState,
}
"""Map from type tags to state classes, used to dispatch when loading from YAML."""

OBJECT_TYPES = {
    # The objects of the Spot mobile manipulation experiment, and their types
    "eraser1": "eraser",
    "board1": "board",
    "black_dresser": "drawer",
    "door1": "door",
    "cabinet1": "surface",
}
"""Map from object names to their type tags."""


def object_state_from_yaml_data(data: Dict[str, Any]) -> Any:
    """Construct the right object state class from YAML data tagged with its type.

    :param data: YAML data including a "type" key
    :return: Constructed object state instance
    :raises KeyError: If the type tag is missing or unregistered
    """
    type_name = data["type"]
    if type_name not in OBJECT_STATE_TYPES:
        raise KeyError(
            f"Unknown object type '{type_name}'. Known types: {sorted(OBJECT_STATE_TYPES)}.",
        )
    return OBJECT_STATE_TYPES[type_name].from_yaml_data(data)
