# DroneAI Labeling Guide

This guide tells you what to mark while you watch a drone video in the labeling GUI, and when to
click. Everyone who labels should follow it, so that the labels mean the same thing no matter who
made them. The AI model learns from these marks, so a wrong or missing mark teaches it the wrong thing.

## The short version

1. Watch the whole video. Mark **every** take-off, landing and crash in it.
2. Click the button **at the moment** the event happens. Pause first if you need to.
3. A crash where the drone keeps flying (or could) is **Minor Crash**. A crash that ends the flight
   or visibly breaks the drone is **Severe Crash**.
4. If you are not sure, pick the best label and write the video and event number in your notes.

## The four events

| Button | Mark it when | Click at |
|---|---|---|
| Take-off | The drone leaves the ground (or a hand) and flies. | The first frame it is clearly off the ground. |
| Land | The pilot brings the drone down on purpose and it settles. | The frame it touches down and stays down. |
| Minor Crash | The drone hits something by accident, but it can fly again. | The frame of impact. |
| Severe Crash | The drone hits something by accident and cannot fly again, or is visibly broken. | The frame of impact. |

These definitions come from the team decision of 2026-09-23.

### Take-off

- Mark the moment the drone lifts off, not when the motors spin up.
- A hand launch counts as a take-off. Mark the moment it leaves the hand.
- If the drone takes off again after a landing or a minor crash, mark that take-off too.
- If the video starts with the drone already in the air, there is no take-off to mark.

### Land

- A landing is controlled. The drone comes down slowly and stays down.
- A small bounce on touchdown is still a landing. Mark the first touch.
- A hand catch at the end of a flight counts as a landing.
- If the drone falls or drops hard out of control, it is a crash, not a landing.
- If the video ends with the drone still in the air, there is no landing to mark.

### Crash (minor or severe)

A crash is accidental contact: with the ground, a wall, a tree, a gate, a person, or anything else.
Mark the moment of impact. Then decide the severity from what happens **after** the impact.

**Minor Crash.** The drone can fly again. Examples:

- It clips a branch or a gate and keeps flying.
- It hits the ground, flips, and the pilot flies off again or picks it up and relaunches it.
- It bumps a wall and recovers in the air.

**Severe Crash.** The drone cannot fly anymore, or you can see it is broken. Examples:

- Parts or propellers break off.
- It ends upside down or stuck (in a tree, in water) and the flight is over.
- The video ends right after the impact and the drone never flies again in the video.

If one flight has two impacts close together (hits a tree, then falls to the ground), mark each
impact that you can see separately. Judge each one by what happens after it.

## When to click

The GUI saves the time of the frame on screen at the moment you click. It then cuts a short clip
around that time. Clicking late moves the event to the edge of the clip, or out of it.

- Aim to be within half a second of the event.
- The easiest way: press **Pause** the moment you see the event, then click the event button.
  The mark uses the paused frame.
- If you missed it, use **Rewind 10s**, then watch the moment again and pause on it.
- Do not click early "just in case". A mark before the event is as wrong as one after it.

## View type: FPV or Third-person

Each video has a view type. It changes what an event looks like on screen.

| View type | What it is | Examples |
|---|---|---|
| FPV | The camera is on the drone. You see what the drone sees. | Goggles recording, onboard DJI Tello camera, FPV simulator view |
| Third-person | The camera watches the drone from outside. You see the drone in the picture. | A phone filming from the ground, a simulator chase camera |

How the events look in **FPV**:

- **Take-off:** the ground drops away and the view starts moving.
- **Land:** the ground comes up, the view settles and stops moving.
- **Crash:** a sudden jolt, spin, or tumble. Often the image freezes, goes black, shows static, or
  ends up sideways or upside down.

How the events look in **Third-person**:

- You can see the drone lift off, touch down, or hit something directly.
- If the drone leaves the frame and you cannot see the event, do not guess. Only mark what you see.

If the view type set for the video is wrong (for example it says FPV but the camera is a phone on
the ground), keep labeling and write down the video ID and the correct view type.

## Source: Real or Simulation

Real videos are real drones. Simulation videos come from a simulator (for example FPV SkyDive,
FPV Freerider, or Steam Drone Simulator). Label both with the same four events.

In a simulator the drone cannot really break, so use this rule for severity:

- The pilot keeps flying after the impact: **Minor Crash**.
- The simulator resets, respawns, or shows a crash screen, or the run ends: **Severe Crash**.
- A reset that the pilot chose (not caused by an impact) is not an event.

## What not to mark

- Hovering, flips, rolls, fast turns, or near misses with no contact.
- The motors starting or stopping without the drone moving.
- Cuts in an edited video. A jump to a new scene is not an event. If the new scene starts with a
  take-off, mark that take-off.
- Anything you cannot see. If the event happens off screen, skip it.

## Why every event matters

Later, the parts of a video with no marks are used as examples of "nothing happening". If you skip
a landing, the model learns that a landing is "nothing happening". So:

- Label the whole video, start to end, not only the first few events.
- If you stop partway through a video, say so, so the video is not treated as fully labeled.

## Fixing mistakes

The GUI has no undo for event marks. If you click the wrong button or click at the wrong time:

1. Do not try to fix it by clicking again.
2. Write down the video ID, the event number shown in "Events Marked", and what it should be
   (the right label, or "delete").
3. Send the list to the reviewer when you finish the video.

## Checklist before you finish a video

- [ ] I watched the video to the end.
- [ ] Every take-off, landing and crash I could see is marked.
- [ ] Each crash is marked minor or severe using what happened after the impact.
- [ ] I wrote down any wrong clicks, unsure labels, or a wrong view type.

## Hard cases

| Situation | Label |
|---|---|
| Hard landing on purpose that bounces, drone is fine | Land |
| Drone drops out of the sky and hits the ground, pilot relaunches it | Minor Crash, then Take-off |
| Drone hits the ground, looks undamaged, and the pilot ends the flight there | Minor Crash (it could fly again) |
| Drone clips a gate in a simulator and keeps flying | Minor Crash |
| Simulator respawns the drone after it hits a wall | Severe Crash |
| Video ends right after an impact and you can't see the drone's state | Severe Crash |
| Split screen or a mix of FPV and third-person shots | Label the events you see; note the mix in your notes |

When a case is not in this table, choose the label that best matches the definitions above, and
write it down so the team can add it here.
