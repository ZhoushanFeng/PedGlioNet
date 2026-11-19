"""Regenerate the five axis-labelled publication SVGs from the accepted vectors."""
import argparse
from pathlib import Path

from lxml import etree


SVG = "{http://www.w3.org/2000/svg}"
SOURCE_DIR = Path(__file__).resolve().parents[1] / "analysis" / "editable_figures"


def label_figure(source, number):
    svg = etree.parse(str(source)).getroot()
    group = svg.xpath('//*[@id="figure_1"]')[0]
    _, _, width, height = map(float, svg.get("viewBox").split())
    left = 0

    def add_label(x, y, text, vertical=False, size=9.1):
        style = f"font-family: 'Times New Roman'; font-size: {size}px; fill: #19242d; text-anchor: middle;"
        node = etree.SubElement(group, SVG + "text", x=str(x), y=str(y), style=style)
        if vertical:
            node.set("transform", f"rotate(-90 {x} {y})")
        node.text = text

    def replace_label(old_text, new_text, expected=1):
        labels = [node for node in group.iter(SVG + "text") if node.text == old_text]
        if len(labels) != expected:
            raise ValueError(f"{source.name}: expected {expected} labels matching {old_text!r}, found {len(labels)}")
        for node in labels:
            node.text = new_text

    if number == 2:
        replace_label("Dice", "Dice score")
        for axis in ("axes_2", "axes_3", "axes_4"):
            svg.xpath('//*[@id=$axis]', axis=axis)[0].set("transform", "translate(0,12)")
        height += 12
        add_label(270, 230, "Tumor region")
        add_label(13, 332, "Tumor region", vertical=True)
        add_label(257, 332, "Held-out source", vertical=True)
        add_label(394, 420, "Tumor region")
        add_label(394, 435, "Color: mean Dice")
    elif number == 3:
        left = -9
        width += 9
        replace_label("Dice difference", "Paired Dice difference", expected=2)
        replace_label("Seed", "Pediatric training seed")
        replace_label("ET Dice contribution", "Contribution to ET Dice difference")
        add_label(6, 141, "Tumor region", vertical=True)
        add_label(270, 141, "Tumor region", vertical=True)
        add_label(4, 354, "Held-out source", vertical=True)
        add_label(269, 354, "Source group", vertical=True)
        add_label(257, 353, "Dice difference", vertical=True)
    elif number == 4:
        left = -4
        width += 4
        for center, edge in [(137.9, 8), (395.75, 265.9)]:
            add_label(center, 30, "Held-out source")
            add_label(center, 193, "Pediatric training seed")
            add_label(edge, 111, "Tumor region", vertical=True)
    elif number == 5:
        replace_label("Cases", "Number of examinations", expected=2)
        replace_label("Mean volume (mm³)", "Mean false-positive ET volume (mm³)")
        replace_label("PPV", "Component PPV")
        for x in (14, 278):
            for y in (110, 278):
                add_label(x, y, "Pediatric training seed", vertical=True)
    elif number == 7:
        left = -22
        width += 22
        # The existing x-axis title uses outlines. Its glyph definitions are reused elsewhere.
        svg.xpath('//*[@id="text_6"]')[0].set("visibility", "hidden")
        add_label(-10, 76, "Tumor region", vertical=True, size=11)
        add_label(201, 169, "Dice difference (development-only minus historical)", size=10.5)
    else:
        raise ValueError(f"No publication axis edits recorded for Figure {number}")

    background = etree.Element(SVG + "rect", x=str(left), y="0", width=str(width), height=str(height), fill="white")
    svg.insert(0, background)
    width_mm = 127 * (width / 360) if number == 7 else 180
    height_mm = width_mm * height / width
    svg.set("width", f"{width_mm}mm")
    svg.set("height", f"{height_mm}mm")
    svg.set("viewBox", f"{left} 0 {width} {height}")
    return etree.tostring(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    for number in (2, 3, 4, 5, 7):
        source, = SOURCE_DIR.glob(f"Figure{number}_*.svg")
        (output / source.name).write_bytes(label_figure(source, number))


if __name__ == "__main__":
    main()
