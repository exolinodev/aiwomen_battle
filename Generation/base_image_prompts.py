"""
Base image prompts for the swimsuit around the world generation workflow.
This file contains the prompts used to generate the base images for animations.
"""

# Base image generation prompts for different countries
BASE_IMAGE_PROMPTS = [
    {
        "id": "01_brazil",
        "text": "Wide-angle photorealistic shot capturing a stunningly beautiful, highly athletic young woman competing in beach volleyball on Copacabana Beach, Rio de Janeiro. She wears a sporty, athletic-cut bikini prominently featuring the green, yellow, and blue design of the Brazilian flag, similar to official team gear. The shot shows the action on the sand court, with the turquoise ocean and Sugarloaf Mountain in the background. She has a fit, powerful physique, caught mid-play (e.g., setting the ball), looking intensely towards the camera direction but breaking into a confident, bright smile showing teeth. The image captures the high-energy, competitive spirit of Brazilian beach sports. Sharp focus on the subject, dynamic action feel, hyperrealistic details, 8K resolution."
    },
    {
        "id": "02_australia",
        "text": "Wide-angle photorealistic shot capturing a stunningly beautiful, athletic young woman, professional surfer style, on Bondi Beach, Sydney. She wears a sporty performance bikini clearly displaying the Australian flag's stars and Union Jack motif, designed for surfing. She stands confidently beside her pro-style surfboard (also subtly Aussie themed). The shot includes the white sand, powerful blue waves, and the distant Opera House/Harbour Bridge. She has a lean, strong physique, slicked-back wet hair, and is flashing a wide, dazzling smile at the camera, showing teeth. The image captures the cool, competitive, and sunny vibe of Australian surf competitions. Sharp focus, clear background, hyperrealistic details, 8K resolution."
    },
    {
        "id": "03_greece",
        "text": "Wide-angle photorealistic shot capturing a stunningly beautiful, athletic young woman taking a break during a stand-up paddleboard race near Santorini, Greece. She wears a sporty, functional bikini designed with the iconic blue and white horizontal stripes and cross of the Greek flag. She's kneeling or sitting on her race paddleboard in the calm turquoise water near the cliffs with white-washed villages. She has a fit, toned physique, looking directly at the camera with a triumphant, bright smile showing teeth. The image captures the blend of intense athletic activity and stunning Greek island scenery. Sharp focus, hyperrealistic water and environment, 8K resolution."
    },
    {
        "id": "04_maldives",
        "text": "Wide-angle photorealistic shot capturing a stunningly beautiful, athletic young woman participating in a water sports event (like jet ski racing or competitive swimming) in the Maldives. She wears a sleek, athletic one-piece or bikini featuring the vibrant red, green, and white crescent design of the Maldivian flag. She's pictured near her jet ski or at the edge of a racing lane marker by an overwater bungalow complex, crystal clear turquoise water all around. She has a fit, streamlined physique, water glistening on her skin, turning to the camera with a vibrant, winning smile showing teeth. The image captures luxury combined with high-octane water sports energy. Sharp focus, hyperrealistic details, 8K resolution."
    },
    {
        "id": "05_hawaii",
        "text": "Wide-angle photorealistic shot capturing a stunningly beautiful, pro-level female surfer on Waikiki Beach, Honolulu, Hawaii. She wears a sporty, competition-style bikini showcasing a bold design inspired by the Hawaiian state flag (red, white, blue stripes and Union Jack canton). She's holding her high-performance surfboard, ready to enter a heat, with Diamond Head crater prominent in the background. She has a strong, athletic build, exuding confidence, and offers a big, genuine 'aloha' smile directly at the camera, showing teeth. The image captures the spirit of elite surfing competition in a world-famous location. Sharp focus, dynamic composition, hyperrealistic details, 8K resolution."
    },
    {
        "id": "06_italy",
        "text": "Wide-angle photorealistic shot capturing a stunningly beautiful, athletic young woman competing in a coastal rowing or kayaking event along the Amalfi Coast, Italy. She wears a sporty, ergonomic rowing suit or bikini designed with the vertical green, white, and red stripes of the Italian flag. She's pictured in her sleek racing kayak/scull near the dramatic cliffs and colorful villages. She has a fit, strong physique, pausing her paddling to look directly at the camera with a radiant, competitive smile showing teeth. The image captures the elegance of the sport combined with the dramatic beauty of the Italian coastline. Sharp focus, clear water reflections, hyperrealistic details, 8K resolution."
    }
]

# Negative prompt to avoid unwanted elements
BASE_IMAGE_NEGATIVE_PROMPT = "ugly, deformed, blurry, low quality, extra limbs, disfigured, poorly drawn face, bad anatomy, cartoon, drawing, illustration, text, watermark, signature, multiple people, nudity, inappropriate content." 