"""Quick test of PEBBLE training loop."""
from src.train.train_pebble import train_pebble
from src.rewards.reacher_rewards import get_reward_function

teacher_fn = get_reward_function('Ra')

print('Starting PEBBLE training test...')

result = train_pebble(
    teacher_fn,
    'Ra',
    seed=0,
    total_steps=1000,
    start_steps=100,
    eval_interval=500,
    eval_episodes=2,
    preference_query_interval=200,
    preference_batch_size=2,
    reward_model_epochs=2,
    output_dir=None,
    evaluation_formulations=['Ra'],
)

print('✓ Training completed')
print(f'✓ Result keys: {list(result.keys())}')
print(f'✓ Evaluation steps: {result["evaluation_steps"]}')
print(f'✓ Preference query steps: {result["preference_query_steps"]}')
print(f'✓ Preference pairs queried: {result["preference_pairs_queried"]}')
print(f'✓ Reward model losses: {len(result["reward_model_losses"])} updates')
print('\nStep 4 (PEBBLE Training Loop) working correctly!')
