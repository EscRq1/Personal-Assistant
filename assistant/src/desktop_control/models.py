from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point(Model):
    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)


class Region(Point):
    width: int = Field(ge=1, strict=True)
    height: int = Field(ge=1, strict=True)


class ObserveRequest(Model):
    region: Region | None = None
    frame_id: str | None = None

    @model_validator(mode="after")
    def region_needs_frame(self):
        if (self.region is None) != (self.frame_id is None):
            raise ValueError("Provide region and frame_id together, or omit both for a full-screen view")
        return self


class Click(Point):
    kind: Literal["click"]
    button: Literal["left", "middle", "right"] = "left"
    clicks: int = Field(default=1, ge=1, le=2)


class Move(Point):
    kind: Literal["move"]


class Drag(Model):
    kind: Literal["drag"]
    path: list[Point] = Field(min_length=2, max_length=500)
    duration: float = Field(default=0.5, ge=0.05, le=10)
    button: Literal["left", "middle", "right"] = "left"


class Scroll(Model):
    kind: Literal["scroll"]
    steps: int = Field(ge=-100, le=100, strict=True)


class Keypress(Model):
    kind: Literal["keypress"]
    keys: list[str] = Field(min_length=1, max_length=6)
    hold: float = Field(default=0.05, ge=0, le=10)


class TypeText(Model):
    kind: Literal["type"]
    text: str = Field(max_length=2000)
    interval: float = Field(default=0.01, ge=0, le=0.05)

    @field_validator("text")
    @classmethod
    def supported_text(cls, value):
        if any(character not in "\n\t" and not 32 <= ord(character) <= 126 for character in value):
            raise ValueError("v1 typing supports printable ASCII, tabs and newlines")
        return value


class Wait(Model):
    kind: Literal["wait"]
    seconds: float = Field(default=0.25, ge=0, le=10)


Action = Annotated[Click | Move | Drag | Scroll | Keypress | TypeText | Wait, Field(discriminator="kind")]


class ActRequest(Model):
    frame_id: str
    action: Action


class Observation(Model):
    frame_id: str
    captured_at: str
    width: int
    height: int
    image_data_url: str
    origin_x: int = 0
    origin_y: int = 0
    screen_width: int
    screen_height: int
    coordinate_space: Literal["image"] = "image"
    view: Literal["full", "region"] = "full"


class ActResult(Model):
    status: Literal["ok"] = "ok"
    observation: Observation
