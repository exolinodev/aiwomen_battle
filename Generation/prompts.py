"""
Prompt templates for image and animation generation.
"""

# Template for generating base images
BASE_IMAGE_TEMPLATE = (
    "Wide-angle photorealistic shot capturing a stunningly beautiful, highly athletic young woman "
    "competing in beach volleyball. She wears a sporty, athletic-cut bikini prominently featuring "
    "the {colors} design of the {flag_description}, similar to official team gear. The shot shows "
    "the action on the sand court, with the turquoise ocean in the background. She has a fit, "
    "powerful physique, caught mid-play (e.g., setting the ball), looking intensely towards the "
    "camera direction but breaking into a confident, bright smile showing teeth. The image captures "
    "the high-energy, competitive spirit of {name} beach sports. Sharp focus on the subject, "
    "dynamic action feel, hyperrealistic details, 8K resolution."
)

# Template for generating animations
ANIMATION_TEMPLATE = (
    "Create a dynamic and cinematic animation of an athlete in motion. The athlete is wearing {colors} colors representing {name}. {flag_description} The movement should be smooth and fluid, with a gentle camera motion that follows the athlete's graceful movements. The lighting should be dramatic and cinematic, enhancing the sense of motion and energy. The background should subtly incorporate elements of {name}'s national identity while maintaining focus on the athlete's movement."
)

# Common negative prompt for both base images and animations
NEGATIVE_PROMPT = (
    "ugly, deformed, blurry, low quality, extra limbs, disfigured, poorly drawn face, "
    "bad anatomy, cartoon, drawing, illustration, text, watermark, signature, multiple people, "
    "nudity, inappropriate content."
) 