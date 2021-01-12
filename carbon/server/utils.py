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
