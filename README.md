

# TV Show Recommendation Engine 🚀📺

## Overview

Welcome aboard the ultimate TV show discovery spaceship! This isn't your ordinary recommendation engine—it's a bespoke journey to your next binge-worthy series, equipped with the powers of machine learning and natural language processing. Plus, get a visual and narrative taste of what's to come with AI-generated posters and scripts for a blend of suggested shows. Ready to find your next obsession?

## Installation

Prepare for lift-off by installing the necessary packages to fuel this engine:

```bash
pip install -r requirements.txt
```

Ensure your Python environment is ready for the journey ahead!

## Configuration

Before we take off, make sure your communications system (API keys) is set up:

1. Add your OpenAI API key to an environment variable:
   ```bash
   export OPENAI_API_KEY='your_secret_api_key_here'
   ```

2. Or keep it secure in a `.env` file in the project's root:
   ```
   OPENAI_API_KEY=your_secret_api_key_here
   ```

## Usage

### Generating Embeddings

Create a flavor profile for each TV show based on their descriptions:

```python
csv_file_path = 'path_to_your_tv_show_data.csv'
embeddings_dict = generate_embeddings(csv_file_path)
```

### Personalized Recommendations

Let our system recommend shows that feel like they were handpicked just for you:

```python
user_favorite_shows = get_user_favorite_shows(list(embeddings_dict.keys()))
recommendations = get_recommendations(user_favorite_shows, embeddings_dict)
```

### Creating a New Show Experience

Merge your recommended shows into a new concept:

```python
show_name_and_description = generate_show_name(user_favorite_shows)
print("Generated new show idea: ", show_name_and_description)
```

### Visual and Script Magic

Generate a script and a visually stunning poster for this newly imagined show:

```python
image_url = generate_image_with_dalle(show_name_and_description)
open_dalle_image(image_url)
```

See how your recommended shows blend into a new exciting series, complete with its unique storyline and poster.

## Features

- **Taste Matcher:** Converts show descriptions into vectors to find your perfect TV match.
- **Personalized Suggestions:** Crafts show recommendations based on your preferences using sophisticated algorithms.
- **Creative Synthesis:** Generates a script and poster for a new show concept derived from your recommendations, providing a glimpse into what could be your next favorite series.
- **Artistic Flair:** Uses state-of-the-art AI to create compelling visuals for the newly imagined show.
- **Logging Lore:** Chronicles system activities and errors to enhance future voyages.

## Contributing

Feel inspired? We welcome your creative contributions! Fork this repository, make your magical modifications, and propel them back to us via a pull request.

## License

This project is freely open-sourced under the MIT License. Details are available in the LICENSE file.

---
