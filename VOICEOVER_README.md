# Voice-Over Integration for Video Generation

This module integrates ElevenLabs text-to-speech technology to generate voice-overs for your videos.

## Setup

1. Make sure you have an ElevenLabs API key. You can get one by signing up at [ElevenLabs](https://elevenlabs.io/).

2. Add your API key to the `Generation/.env` file:
   ```
   ELEVENLABS_API_SECRET=your_api_key_here
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Testing the Integration

Run the test script to verify that the ElevenLabs integration is working correctly:

```
python test_voiceover.py
```

This will:
1. Test the connection to the ElevenLabs API
2. Generate a simple test voice-over
3. Play the generated audio

## Using Voice-Over in Your Videos

The voice-over integration is now part of the main workflow in `run_with_params.py`. The workflow:

1. Generates animation clips
2. Creates a voice-over with the specified script
3. Combines the video clips with the voice-over to create the final video

### Customizing the Voice-Over

To customize the voice-over, you can modify the `generate_voiceover` function in `Generation/voiceover.py`. The current script format is:

```
"Video #{video_number}. What country do you like most? {country1} from the image generation, then {country2}. Don't forget to like and subscribe to our channel for more amazing content!"
```

You can change this format to match your needs.

### Available Voices

The default voice is "Bella", but you can use any voice available in your ElevenLabs account. To change the voice, modify the `voice` parameter in the `generate_voiceover` function.

### Adding Background Music

The video editor supports adding background music to your videos. Place your background music file in the `music` directory and it will be automatically included in the final video.

## Troubleshooting

If you encounter issues with the voice-over generation:

1. Check that your ElevenLabs API key is correct in the `.env` file
2. Verify that you have sufficient credits in your ElevenLabs account
3. Check the console output for error messages

## Advanced Usage

For more advanced usage, you can directly use the functions in `Generation/voiceover.py` and `Generation/video_editor.py` in your own scripts. 