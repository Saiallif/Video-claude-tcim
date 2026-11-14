"""Official results - TCIM Sdn Bhd Bowling Night, 13/11/2026.

Transcribed from the printed league sheets (names are exactly as printed by the
scoring system, so some are abbreviated). Every row is checked: game 1 + game 2 == total.
"""

# (rank, team, game1, game2, total)
TEAMS = [
    (1, "TEAM 7", 538, 576, 1114),
    (2, "TEAM 3", 524, 524, 1048),
    (3, "TEAM 8", 512, 525, 1037),
    (4, "TEAM 2", 460, 554, 1014),
    (5, "TEAM 9", 524, 473, 997),
    (6, "TEAM 11", 488, 499, 987),
    (7, "TEAM 4", 476, 505, 981),
    (8, "TEAM 10", 451, 428, 879),
    (9, "TEAM 5", 404, 471, 875),
    (10, "TEAM 1", 417, 450, 867),
    (11, "TEAM 6", 400, 435, 835),
    (12, "TEAM 12", 408, 333, 741),
]

# (rank, name, average, game1, game2, total)   - sheet "Men", page 1 of 2
MEN = [
    (1, "Rudy", 143, 129, 157, 286), (2, "Azam", 131, 126, 137, 263),
    (3, "Faizl", 127, 126, 128, 254), (4, "Thqif", 126, 137, 115, 252),
    (5, "Steve", 125, 139, 111, 250), (6, "Napi", 123, 97, 150, 247),
    (7, "Shaz", 122, 129, 115, 244), (8, "Rosdi", 117, 115, 120, 235),
    (9, "Wi", 115, 91, 139, 230), (10, "Asyrf", 114, 107, 121, 228),
    (11, "Erick", 109, 113, 106, 219), (12, "Nizam", 109, 96, 122, 218),
    (13, "Peter", 107, 99, 116, 215), (14, "Dnish", 107, 95, 120, 215),
    (15, "Henry", 107, 112, 103, 215), (16, "Farid", 104, 91, 118, 209),
    (17, "Rahim", 104, 115, 93, 208), (18, "Iqbal", 104, 103, 105, 208),
    (19, "Kslee", 103, 119, 88, 207), (20, "Alif", 102, 105, 99, 204),
    (21, "Shah", 102, 82, 122, 204), (22, "Rsyad", 100, 96, 104, 200),
    (23, "Fahmi", 98, 81, 116, 197), (24, "Jf", 97, 90, 105, 195),
    (25, "Maell", 96, 89, 104, 193), (26, "Swjje", 95, 100, 90, 190),
    (27, "Faisal", 93, 100, 87, 187), (28, "Jimmy", 90, 94, 86, 180),
    (29, "Uddin", 90, 69, 111, 180), (30, "Azim", 88, 91, 86, 177),
    (31, "Rmizu", 87, 108, 67, 175), (32, "Ammar", 86, 103, 70, 173),
    (33, "Clvin", 83, 83, 84, 167), (34, "Ibrhm", 82, 87, 78, 165),
    (35, "Emir", 82, 85, 80, 165), (36, "Shful", 81, 93, 70, 163),
    (37, "Tvesh", 79, 86, 72, 158), (38, "Zeno", 71, 61, 81, 142),
    (39, "Nopal", 70, 52, 88, 140), (40, "Azrul", 69, 81, 57, 138),
    (41, "Dev", 66, 43, 89, 132), (42, "Don", 63, 61, 65, 126),
    (43, "Aiman", 62, 67, 58, 125),
]

# sheet "Women"
WOMEN = [
    (1, "Aisha", 112, 100, 125, 225), (2, "Josep", 97, 85, 109, 194),
    (3, "Alysa", 95, 102, 89, 191), (4, "Hajar", 94, 93, 95, 188),
    (5, "Klsom", 90, 78, 103, 181), (6, "Aina", 90, 93, 87, 180),
    (7, "Nbila", 89, 96, 82, 178), (8, "Wani", 89, 92, 86, 178),
    (9, "Najwa", 87, 75, 99, 174), (10, "Syfqh", 82, 91, 73, 164),
    (11, "Tira", 78, 73, 83, 156), (12, "Kina", 77, 88, 67, 155),
    (13, "Penny", 77, 90, 64, 154), (14, "Shkin", 75, 84, 66, 150),
    (15, "Ain", 72, 88, 57, 145), (16, "Lis", 71, 65, 78, 143),
    (17, "Wana", 70, 63, 77, 140),
]

# Official "Overall" boxes printed on the team sheet
AWARDS = {
    "team_game": [("TEAM 7", 576), ("TEAM 2", 554), ("TEAM 8", 525)],
    "team_series": [("TEAM 7", 1114), ("TEAM 3", 1048), ("TEAM 8", 1037)],
    "men_game": [("RUDY", 157), ("NAPI", 150), ("STEVE", 139)],
    "men_series": [("RUDY", 286), ("AZAM", 263), ("FAIZL", 254)],
    "women_game": [("AISHA", 125), ("JOSEP", 109), ("KLSOM", 103)],
    "women_series": [("AISHA", 225), ("JOSEP", 194), ("ALYSA", 191)],
}
STAR_BOWLER = ("AIN", 16, 88)  # 16 pins over average with an 88 game (women's sheet)


def _check():
    for r, t, g1, g2, tot in TEAMS:
        assert g1 + g2 == tot, ("team", t)
    for r, n, a, g1, g2, tot in MEN + WOMEN:
        assert g1 + g2 == tot, ("player", n)
    assert [t[4] for t in TEAMS] == sorted([t[4] for t in TEAMS], reverse=True)
    assert [m[5] for m in MEN] == sorted([m[5] for m in MEN], reverse=True)
    assert [w[5] for w in WOMEN] == sorted([w[5] for w in WOMEN], reverse=True)
    return True


if __name__ == "__main__":
    print("results data OK:", _check(), len(TEAMS), len(MEN), len(WOMEN))
