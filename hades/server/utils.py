from pydantic import BaseModel


class ImmutBaseModel(BaseModel):
    class Config:
        # faux immutability of fields
        allow_mutation = False
        # validate field defaults
        validate_all = True


class AssignValidateBaseModel(BaseModel):
    class Config:
        # whether to perform validation on assignment to attributes
        validate_assignment = True


class StrictBaseModel(BaseModel):
    class Config:
        allow_mutation = False
        validate_all = True
        # whether to ignore, allow, or forbid extra attributes during model initialization.
        extra = "forbid"


def printBox(text):
    """Print CLI Box around given text string"""
    times = 4
    boxVertical = "│"
    boxHorizontal = "─"
    boxTopLeft = "┌%s" % (boxHorizontal * times)
    boxTopRight = "%s┐" % (boxHorizontal * times)
    boxBottomLeft = "└%s" % (boxHorizontal * times)
    boxBottomRight = "%s┘" % (boxHorizontal * times)
    boxMiddleLeft = "%s%s" % (boxVertical, " " * times)
    boxMiddleRight = "%s%s" % (" " * times, boxVertical)
    lentext = len(text)
    print("%s%s%s" % (boxTopLeft, boxHorizontal * lentext, boxTopRight))
    print("%s%s%s" % (boxMiddleLeft, text, boxMiddleRight))
    print("%s%s%s" % (boxBottomLeft, boxHorizontal * lentext, boxBottomRight))
