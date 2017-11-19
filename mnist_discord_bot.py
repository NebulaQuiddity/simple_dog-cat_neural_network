import discord
#import feed_forward_neural_network as ffnn

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))

    async def on_message(self, message):
        if message.author == self.user:
            return

        elif message.content.startswith('$hello'):
            await self.send_message(message.channel, 'Hello ' + (str(message.author)[:-5]) + '!')

        print('Message from {0.author}: {0.content}'.format(message))

client = MyClient()
client.run('MzgxOTAxNTE5NjcyMzExODA5.DPN5GQ.IKSfQZc9iekbnPB2fcQ1Zcq_ujc')

