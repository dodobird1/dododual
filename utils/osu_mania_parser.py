"""
osu!mania Beatmap Parser

A module that converts osu!mania beatmaps into Python objects.
"""

from __future__ import annotations
import os
from typing import List, Optional, Literal
from dataclasses import dataclass, field


# Type alias for hit sounds
HitSound = List[Literal['normal', 'whistle', 'finish', 'clap']]


@dataclass
class TimingPoint:
    """
    Represents a timing point in an osu! beatmap.
    """
    # Start time of the timing section, in milliseconds from the beginning of the beatmap's audio
    # The end of the timing section is the next timing point's time, or never if this is the last timing point
    time: int
    # Slider Velocity as a multiplier
    velocity: float
    # Amount of beats in a measure. Inherited timing points ignore this property.
    timing_signature: int
    # Default sample set for hit objects.
    # 0 = beatmap default, 1 = normal, 2 = soft, 3 = drum
    sample_set: Literal[0, 1, 2, 3]
    # Custom sample index for hit objects. 0 indicates osu!'s default hitsounds.
    sample_index: int
    # Volume percentage for hit objects
    volume: int
    # Whether or not the timing point is uninherited
    uninherited: bool
    # Whether or not kiai time is enabled
    kiai_time: bool
    # Whether or not the first barline is omitted in osu!mania
    omit_first_bar_line: bool
    # The duration of a beat, in milliseconds.
    bpm: Optional[int] = None

    @staticmethod
    def parse(line: str) -> TimingPoint:
        """Parse a timing point from a line in the beatmap file."""
        members = line.split(',')
        beat_length = float(members[1])
        bpm = 0
        velocity = 1.0

        if beat_length > 0:
            bpm = round(60000 / beat_length)
        else:
            velocity = abs(100 / beat_length)

        effects = int(members[7])

        return TimingPoint(
            time=int(members[0]),
            bpm=bpm if bpm != 0 else None,
            velocity=velocity,
            timing_signature=int(members[2]),
            sample_set=int(members[3]),  # type: ignore
            sample_index=int(members[4]),
            volume=int(members[5]),
            uninherited=(members[6] == '1'),
            kiai_time=((effects & 0b1) != 0),
            omit_first_bar_line=((effects & 0b100) != 0)
        )


@dataclass
class HitObject:
    """
    Represents a hit object in an osu!mania beatmap.
    """
    type: Literal['note', 'hold']
    hit_sound: HitSound
    new_combo: bool
    combo_colors_skipped: int
    # Position in osu! pixels of the object.
    x: int
    # Position in osu! pixels of the object.
    y: int
    # Time when the object is to be hit, in milliseconds from the beginning of the beatmap's audio.
    time: int
    # End time of the hold, in milliseconds from the beginning of the beatmap's audio.
    end_time: int

    @staticmethod
    def parse(line: str) -> HitObject:
        """Parse a hit object from a line in the beatmap file."""
        members = line.split(',')
        type_flags = int(members[3])

        note = (type_flags & 0b1) != 0
        hold = (type_flags & 0b10000000) != 0

        new_combo = (type_flags & 0b100) != 0
        combo_colors_skipped = (type_flags & 0b11100) // 4

        hitsound_flags = int(members[4])
        hitsounds: HitSound = []

        if (hitsound_flags & 0b1) != 0:
            hitsounds.append('normal')
        if (hitsound_flags & 0b10) != 0:
            hitsounds.append('whistle')
        if (hitsound_flags & 0b100) != 0:
            hitsounds.append('finish')
        if (hitsound_flags & 0b1000) != 0:
            hitsounds.append('clap')

        if len(hitsounds) == 0:
            hitsounds.append('normal')

        if note:
            return HitObject(
                type='note',
                hit_sound=hitsounds,
                new_combo=new_combo,
                combo_colors_skipped=combo_colors_skipped,
                x=int(members[0]),
                y=int(members[1]),
                time=int(members[2]),
                end_time=int(members[2])
            )
        elif hold:
            return HitObject(
                type='hold',
                hit_sound=hitsounds,
                new_combo=new_combo,
                combo_colors_skipped=combo_colors_skipped,
                x=int(members[0]),
                y=int(members[1]),
                time=int(members[2]),
                end_time=int(members[5].split(':')[0])
            )
        else:
            raise ValueError("Unknown hit object type!")


@dataclass
class Beatmap:
    """
    A parsed osu!mania beatmap.
    """
    # Romanized Song Title
    title: str = ""
    # Romanized Artist Name
    artist: str = ""
    # Beatmap Creator Name
    creator: str = ""
    # Difficulty Name
    version: str = ""
    # Song Source
    source: str = ""
    # Song Tags split in an array
    tags: List[str] = field(default_factory=list)
    # Beatmap ID
    map_id: int = 0
    # Beatmap Set ID
    mapset_id: int = 0
    # Preview Time offset in milliseconds after audio start
    preview_time: int = 0
    # Key Count of beatmap
    key_count: int = 0
    # HP Drain Rate of beatmap
    hp_drain: float = 0.0
    # Overall Difficulty of beatmap
    difficulty: float = 0.0
    # x positions of each key, arranged in ascending order
    key_positions: List[int] = field(default_factory=list)
    # Slowest bpm of the beatmap
    min_bpm: int = 0
    # Fastest bpm of the beatmap
    max_bpm: int = 0
    # Number of notes in the beatmap
    nb_notes: int = 0
    # Number of hold notes in the beatmap
    nb_holds: int = 0
    # List of timing points in the beatmap
    timing_points: List[TimingPoint] = field(default_factory=list)
    # List of hit objects in the beatmap
    hit_objects: List[HitObject] = field(default_factory=list)

    def get_timing_point(self, time: int) -> TimingPoint:
        """Get the timing point active at the given time."""
        for i in range(len(self.timing_points) - 1, -1, -1):
            if self.timing_points[i].time <= time:
                return self.timing_points[i]
        return self.timing_points[0]

    def add_timing_point(self, line: str) -> None:
        """Add a timing point from a line in the beatmap file."""
        timing_point = TimingPoint.parse(line)

        bpm = timing_point.bpm if timing_point.bpm is not None else 0

        if self.min_bpm == 0:
            self.min_bpm = bpm
        elif timing_point.bpm is not None and self.min_bpm > timing_point.bpm:
            self.min_bpm = bpm

        if self.max_bpm == 0:
            self.max_bpm = bpm
        elif timing_point.bpm is not None and self.max_bpm < timing_point.bpm:
            self.max_bpm = bpm

        self.timing_points.append(timing_point)

    def add_hit_object(self, line: str) -> None:
        """Add a hit object from a line in the beatmap file."""
        hit_object = HitObject.parse(line)

        if hit_object.type == 'note':
            self.nb_notes += 1
        elif hit_object.type == 'hold':
            self.nb_holds += 1

        if hit_object.x not in self.key_positions:
            self.key_positions.append(hit_object.x)

        self.hit_objects.append(hit_object)


def parse_file_sync(path: str) -> Beatmap:
    """
    Parses an osu!mania beatmap file into a Python object.

    Args:
        path: The file path of the osu!mania beatmap

    Returns:
        A Beatmap object containing the parsed data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the beatmap's game mode is not osu!mania
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File at {path} does not exist!")

    beatmap = Beatmap()
    section_reg = r'^\[([a-zA-Z0-9]+)\]$'

    import re

    with open(path, 'r', encoding='utf-8') as f:
        contents = f.read()

    # Normalize line endings (handle both Windows \r\n and Unix \n) and filter out comments and empty lines
    lines = [
        line for line in contents.replace('\r\n', '\n').split('\n')
        if not line.startswith('//') and line.strip() != ''
    ]

    section_to_read = ""

    for line in lines:
        match = re.match(section_reg, line)
        if match is not None:
            section_to_read = match.group(1)
        else:
            if section_to_read == "General":
                gen = line.split(": ")
                gen.append('')
                if gen[0] == "Mode":
                    if gen[1] != '3':
                        raise ValueError("Beatmap's game mode is not set to osu!mania.")
                elif gen[0] == "PreviewTime":
                    beatmap.preview_time = int(gen[1])

            elif section_to_read == "Metadata":
                mdata = line.split(":")
                if mdata[0] == "Title":
                    beatmap.title = mdata[1]
                elif mdata[0] == "Artist":
                    beatmap.artist = mdata[1]
                elif mdata[0] == "Creator":
                    beatmap.creator = mdata[1]
                elif mdata[0] == "Version":
                    beatmap.version = mdata[1]
                elif mdata[0] == "Tags":
                    beatmap.tags = mdata[1].split(' ')
                elif mdata[0] == "BeatmapID":
                    beatmap.map_id = int(mdata[1])
                elif mdata[0] == "BeatmapSetID":
                    beatmap.mapset_id = int(mdata[1])

            elif section_to_read == "Difficulty":
                diff = line.split(":")
                diff.append('')
                if diff[0] == "HPDrainRate":
                    beatmap.hp_drain = float(diff[1])
                elif diff[0] == "CircleSize":
                    beatmap.key_count = int(diff[1])
                elif diff[0] == "OverallDifficulty":
                    beatmap.difficulty = float(diff[1])

            elif section_to_read == "TimingPoints":
                beatmap.add_timing_point(line)

            elif section_to_read == "HitObjects":
                beatmap.add_hit_object(line)

    beatmap.key_positions.sort()

    return beatmap


# For convenience, also provide a function alias matching Python naming conventions
parse_beatmap = parse_file_sync


if __name__ == "__main__":
    import sys
    import json
    from dataclasses import asdict

    if len(sys.argv) < 2:
        print("Usage: python osu_mania_parser.py <beatmap_file>")
        sys.exit(1)

    beatmap = parse_file_sync(sys.argv[1])

    # Convert to dict for JSON serialization
    result = asdict(beatmap)
    print(json.dumps(result, indent=2))
