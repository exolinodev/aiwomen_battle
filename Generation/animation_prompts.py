"""
Animation prompts for the gym short generation workflow.
This file contains the prompts used to generate animation clips from a base image.
"""

# Animation prompts for different scenes in the gym short
ANIMATION_PROMPTS = [
    {
        "id": "01_hold_breath",
        "text": "Subject holds the low squat position, muscles tense under the weight. Subtle breathing motion visible in her chest and shoulders. Slight tremble in her legs indicating effort. Locked camera. Live action style."
    },
    {
        "id": "02_squat_ascend_start",
        "text": "The woman slowly begins to push upwards out of the squat, driving powerfully through her heels. Initial visible strain and muscle engagement in quads and glutes. Camera remains steady, focused on her form."
    },
    {
        "id": "03_pan_up_legs",
        "text": "Camera slowly pans upwards while she holds the low squat, starting focused on her ankles/shoes and smoothly travelling up her taut leggings over her calves and thighs towards her hips. Highlights muscle definition. Subject remains mostly still, focused."
    },
    {
        "id": "04_focus_shift",
        "text": "While holding the squat or during a slow ascent, the subject subtly shifts her gaze slightly, maintaining intense concentration. A bead of sweat might glisten. Handheld camera feel with minimal, natural sway."
    },
    {
        "id": "05_weight_adjust",
        "text": "The subject makes a tiny, controlled adjustment to her grip on the weights or slightly shifts the barbell position (if applicable), maintaining balance and form. Muscles in arms and shoulders flex momentarily. Close follow camera."
    },
    {
        "id": "06_zoom_out_reveal",
        "text": "Subject completes one squat rep, pausing briefly at the top or bottom. Camera smoothly zooms out, revealing more of the modern gym environment – racks, other equipment, bright lights – contextualizing her workout. Cinematic style."
    }
] 