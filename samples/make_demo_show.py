"""Builds samples/demo_show/: a short, ORIGINAL, fictional microdrama (not Kuku content)
in Kuku's screenplay format, with small format quirks on purpose (a different episode
header, an SFX label, a colon outside the asterisks) to rehearse running on a new show.

    python samples/make_demo_show.py
"""
from pathlib import Path

OUT = Path(__file__).parent / "demo_show"

# (kind, label, text, seconds). Camera shots are written around each beat automatically.
EPISODES = [
    ("The Night Shift Heiress", "Mara, a hospital night cleaner, is publicly fired by her stepsister on the "
     "night a dying billionaire asks for her by name.", [
        ("action", None, "Mara mops the marble lobby of Halden Hospital while the night staff ignore her.", 4),
        ("line", "VIVIENNE (Cold, Amused)", "You missed a spot. You always miss a spot.", 3),
        ("line", "MARA (Quiet)", "I'm almost done, Vivienne.", 2),
        ("action", None, "Vivienne kicks the bucket over. Dirty water floods across Mara's shoes.", 3),
        ("sfx", "SOUND FX", "A metal bucket clangs and rolls across the marble.", 2),
        ("line", "VIVIENNE (Gloating)", "Now you are done. You're fired. Security will walk you out.", 4),
        ("line", "MARA (Shocked)", "You can't fire me. You don't even work here!", 3),
        ("line", "VIVIENNE (Smug)", "My fiance owns this hospital. Tonight, so do I.", 4),
        ("action", None, "Two guards grab Mara by the arms as night staff film on their phones.", 4),
        ("line", "MARA (Desperate)", "Please, my mother's medical bills...", 3),
        ("line", "VIVIENNE (Cruel)", "Your mother should have raised a daughter who knows her place.", 4),
        ("action", None, "The lift doors open. A nurse runs out, breathless, holding a patient chart.", 3),
        ("line", "NURSE (Urgent)", "Mara Quinn? Is there a Mara Quinn here?", 3),
        ("line", "NURSE (Urgent)", "Mr. Halden is dying. He is asking for you. Only you.", 4),
        ("action", None, "The guards freeze. Vivienne's smile drops. Every phone turns to Mara.", 3),
        ("line", "VIVIENNE (Stunned)", "Why would Arthur Halden ask for a cleaner?", 3),
        ("sfx", "SOUND FX", "A low cello note swells and cuts to silence.", 3),
    ]),
    ("The Last Request", "At the billionaire's bedside, Mara learns he knew her mother, and Vivienne's fiance "
     "arrives with a lawyer to stop a new will.", [
        ("action", None, "Mara steps into a dim private suite. Arthur Halden lies pale beneath the monitors.", 4),
        ("line", "ARTHUR (Weak, Tender)", "You have her eyes. Your mother's eyes.", 3),
        ("line", "MARA (Confused)", "You knew my mother?", 2),
        ("line", "ARTHUR (Weak)", "I did more than know her, Mara. I failed her.", 4),
        ("action", None, "Arthur presses a sealed envelope into Mara's trembling hands.", 3),
        ("line", "ARTHUR (Urgent)", "Do not open this until the lawyer arrives. Promise me.", 4),
        ("sfx", "SFX", "The suite door bangs open.", 1),
        ("action", None, "Damien Cross, Vivienne's fiance, strides in with a lawyer and two guards.", 3),
        ("line", "DAMIEN (Furious)", "Get the cleaner away from him. Now!", 3),
        ("line", "ARTHUR (Gasping)", "Damien... you will not touch my will.", 3),
        ("line", "DAMIEN (Cold)", "Your will is already signed. Everything goes to me.", 4),
        ("line", "LAWYER (Nervous)", "Sir, there is a problem. He changed it an hour ago.", 4),
        ("line", "DAMIEN (Dangerous)", "Changed it to whom?", 2),
        ("action", None, "Every eye in the room turns slowly to the envelope in Mara's hands.", 3),
        ("sfx", "SOUND FX", "The heart monitor flatlines into a single long tone.", 3),
    ]),
    ("The Envelope", "Damien tries to buy the envelope from Mara, then threatens her mother when she refuses.", [
        ("action", None, "Rain hammers the hospital car park. Mara clutches the envelope under her coat.", 4),
        ("line", "DAMIEN (Smooth)", "Ten thousand dollars. Hand it over and walk away.", 4),
        ("line", "MARA (Defiant)", "He told me to wait for the lawyer.", 3),
        ("line", "DAMIEN (Smooth)", "Fifty thousand. Enough for your mother's surgery.", 4),
        ("action", None, "Mara hesitates, looking at the hospital windows above.", 3),
        ("line", "MARA (Firm)", "No. Whatever is in here, he wanted me to have it.", 4),
        ("line", "DAMIEN (Menacing)", "Then let me tell you what is in room 304.", 4),
        ("line", "MARA (Horrified)", "That's my mother's room.", 2),
        ("line", "DAMIEN (Cold)", "Her oxygen is on my hospital's bill. Bills get cancelled.", 4),
        ("action", None, "Mara rips open the envelope. A single photograph falls into a puddle.", 3),
        ("action", None, "The photo shows a young Arthur Halden holding a baby wrapped in a hospital blanket.", 3),
        ("line", "MARA (Whispering)", "Who is this baby?", 2),
        ("line", "DAMIEN (Pale)", "Give me that photo.", 2),
        ("sfx", "SOUND FX", "Thunder cracks as the frame holds on Mara's face. Cut to black.", 3),
    ]),
]


def ts(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def write_episode(n: int, title: str, logline: str, beats: list) -> str:
    t, blocks = 0.0, []
    for i, (kind, label, text, dur) in enumerate(beats):
        cam_end = t + dur
        blocks.append(f"**CAMERA**:\nStart: ({ts(t)}) | End: ({ts(cam_end)})\n"
                      f"{'Close-up' if i % 2 else 'Medium shot'} framing the moment.\n")
        start, end = t + 0.2, cam_end - 0.1
        if kind == "action":
            blocks.append(f"**ACTION:**\nStart: ({ts(start)}) | End: ({ts(end)})\n{text}\n")
        elif kind == "sfx":
            blocks.append(f"**{label}:**\nStart: ({ts(start)}) | End: ({ts(end)})\n{text}\n")
        else:
            blocks.append(f"**{label}:**\nStart: ({ts(start)}) | End: ({ts(end)})\n\"{text}\"\n")
        t = cam_end
    head = (f"EPISODE NUMBER: {n}\nEPISODE LOGLINE: {logline}\nCONTINUITY NOTES: Demo show '{title}'. "
            f"Original fictional content for testing, not Kuku's.\n\n--- SCRIPT START ---\n\n"
            f"SCENE 01\nScene heading: INT. HALDEN HOSPITAL – NIGHT\nScene start: 00:00.00\n"
            f"Scene end: {ts(t)}\n\nActions and dialogue:\n\n")
    return head + "\n".join(blocks) + "\n--- SCRIPT END ---\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n, (title, logline, beats) in enumerate(EPISODES, 1):
        (OUT / f"ep{n:02d}.txt").write_text(write_episode(n, title, logline, beats), encoding="utf-8")
    print(f"Wrote {len(EPISODES)} episodes to {OUT}")
