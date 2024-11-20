from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Match:
    points: list
    color: str
    tag: str
    confidence: float

    def __str__(self):
        points = [Point(round(point.x, 2), round(point.y, 2)) for point in self.points]
        return f"{self.color} {self.tag}, Confidence: {self.confidence : .2f}, Points: {points}"

    # Default sort by confidence
    def __lt__(self, other):
        return self.confidence < other.confidence


def is_overlap(rect1: Match, rect2: Match) -> bool:
    # Calculate the bounding box of rect1
    r1_x_coords = [point.x for point in rect1.points]
    r1_y_coords = [point.y for point in rect1.points]
    r1_min_x, r1_max_x = min(r1_x_coords), max(r1_x_coords)
    r1_min_y, r1_max_y = min(r1_y_coords), max(r1_y_coords)

    # Calculate the bounding box of rect2
    r2_x_coords = [point.x for point in rect2.points]
    r2_y_coords = [point.y for point in rect2.points]
    r2_min_x, r2_max_x = min(r2_x_coords), max(r2_x_coords)
    r2_min_y, r2_max_y = min(r2_y_coords), max(r2_y_coords)

    # Check if the bounding boxes overlap
    return (r1_max_x >= r2_min_x and r2_max_x >= r1_min_x) and (
        r1_max_y >= r2_min_y and r2_max_y >= r1_min_y
    )


def merge_match(rect1: Match, rect2: Match) -> Match:
    # If they are not same tag and color, throw an error
    if rect1.color != rect2.color or rect1.tag != rect2.tag:
        raise ValueError("Rectangles must have same color and tag to merge")

    x_coords = [p.x for p in rect1.points] + [p.x for p in rect2.points]
    y_coords = [p.y for p in rect1.points] + [p.y for p in rect2.points]

    # Compute new boundary coordinates (min/max x and y)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Return the merged rectangle
    return Match(
        [
            Point(min_x, min_y),  # Bottom-left
            Point(min_x, max_y),  # Top-left
            Point(max_x, max_y),  # Top-right
            Point(max_x, min_y),  # Bottom-right
        ],
        rect1.color,
        rect1.tag,
        max(rect1.confidence, rect2.confidence),
    )


def mergeListOfMatches(boxes: List[Match]) -> List[Match]:
    merged_boxes = []
    for box in boxes:
        for j in range(len(merged_boxes)):
            # Merge the boxes if they overlap
            if (
                is_overlap(box, merged_boxes[j])
                and (box.color == merged_boxes[j].color)
                and (box.tag == merged_boxes[j].tag)
            ):
                merged_boxes[j] = merge_match(box, merged_boxes[j])
                break
        else:
            merged_boxes.append(box)

    return merged_boxes


if __name__ == "__main__":
    # Test the mergeListOfMatches function
    match = Match(
        [Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)], "red", "Sentry", 0.9
    )
    print("First match:", match)
    match2 = Match(
        [Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)], "red", "Sentry", 0.95
    )
    print("Second match:", match2)
    merged = mergeListOfMatches([match, match2])
    print("Merged matches:", merged)
