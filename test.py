from schnapsen.game import SchnapsenGamePlayEngine; from schnapsen.bots import RdeepBot, RandBot; import random; engine = SchnapsenGamePlayEngine(); rng = random.Random(42); bot1 = RdeepBot(num_samples=8, depth=4, rand=rng); bot2 = RandBot(rng); print("Playing a test game..."); winner, points, score = engine.play_game(bot1, bot2, rng); print(f"Winner: Bot {1 if winner == bot1 else 2}"); print(f"Points: {points}"); print(f"Score: {score}")
