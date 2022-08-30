_HUMAN_PN_TO_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_PC_TO_FLAT_HUMAN_PN = "C Db D Eb E F Gb G Ab A Bb B".split()

_PC_TO_SHARP_HUMAN_PN = "C C# D D# E F F# G G# A A# B".split()


class PitchClass(int):
    def __new__(cls, pc):
        if not isinstance(pc, int):
            raise TypeError()
        if pc < 0 or pc >= 12:
            raise ValueError()
        return super().__new__(cls, pc)

    def as_human_pitch_name(self, enharmonics="b"):
        if enharmonics == "b":
            d = _PC_TO_FLAT_HUMAN_PN
        elif enharmonics == "#":
            d = _PC_TO_SHARP_HUMAN_PN
        else:
            raise ValueError()
        return HumanPitchName(d[self])

    def as_lily_pitch_name(self, enharmonics="es"):
        if enharmonics == "es":
            enharmonics = "b"
        elif enharmonics == "is":
            enharmonics = "#"
        else:
            raise ValueError()
        return self.as_human_pitch_name(enharmonics=enharmonics).as_lily_pitch_name()


class PitchInterval(int):
    def __new__(cls, pi):
        if not isinstance(pi, int):
            raise TypeError()
        return super().__new__(cls, pi)


class HumanPitchName(str):
    def __new__(cls, pn):
        if not isinstance(pn, str):
            raise TypeError()
        if len(pn) == 0:
            raise ValueError()
        if pn[0] not in _HUMAN_PN_TO_PC:
            raise ValueError()
        accidentals = list(pn[1:])
        if any(a not in ["b", "#"] for a in accidentals):
            raise ValueError()
        if len(set(accidentals)) > 1:
            raise ValueError()
        if len(accidentals) > 2:
            raise ValueError()
        return super().__new__(cls, pn)

    def as_pitch_class(self):
        pc = _HUMAN_PN_TO_PC[self[0]]
        accidental = 0
        for c in self[1:]:
            if c == "b":
                accidental -= 1
            elif c == "#":
                accidental += 1
            else:
                assert False
        return PitchClass((pc + accidental) % 12)

    def as_lily_pitch_name(self):
        pn = self[0].lower()
        for c in self[1:]:
            if c == "b":
                pn += "es"
            elif c == "#":
                pn += "is"
            else:
                assert False
        return LilyPitchName(pn)


class LilyPitchName(str):
    def __new__(cls, pn):
        if not isinstance(pn, str):
            raise TypeError()
        if len(pn) == 0:
            raise ValueError()
        if pn.lower() != pn:
            raise ValueError()
        if pn[0].upper() not in _HUMAN_PN_TO_PC:
            raise ValueError()
        accidentals = pn[1:]
        if len(accidentals) % 2 != 0:
            raise ValueError()
        accidentals = [accidentals[i : i + 2] for i in range(0, len(accidentals), 2)]
        if any(a not in ["es", "is"] for a in accidentals):
            raise ValueError()
        if len(set(accidentals)) > 1:
            raise ValueError()
        if len(accidentals) > 3:
            raise ValueError()
        return super().__new__(cls, pn)

    def as_pitch_class(self):
        return self.as_human_pitch_name().as_pitch_class()

    def as_human_pitch_name(self):
        pn = self[0].upper()
        for i in range(1, len(self), 2):
            if self[i : i + 2] == "es":
                pn += "b"
            elif self[i : i + 2] == "is":
                pn += "#"
            else:
                assert False
        return HumanPitchName(pn)
