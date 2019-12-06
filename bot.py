import asyncio
import bs4
import discord
import dotenv
import json
import os
import random
import re
import requests

"""
with open("C:/Users/Usuario/Downloads/discord-bot/bot.py") as file:
    exec(file.read())
"""

# Load environment
env_path = "C:/Users/Usuario/Downloads/discord-bot/.env"
dotenv.load_dotenv(env_path)

# ---------- Environment variables ---------- #

prefix = os.getenv("prefix")
token = os.getenv("token")

# ---------- Class helpers ---------- #

def get_raw_guilds():
    """ Get guilds in which the bot exists. """
    url = "https://discordapp.com/api/v6/users/@me/guilds"
    headers = {"Authorization": f"Bot {token}"}
    response = requests.get(url, headers=headers)
    raw_guilds = json.loads(response.content)

    return raw_guilds

# ---------- Classes ---------- #

class Command:
    """ Describe a command available from bot. """
    def __init__(self, name, description, example):
        self.name = name
        self.description = description
        self.example = example

    def get_brief_data(self):
        return f"{self.name}: {self.description}"

    def get_extended_data(self):
        return (f"Description: \n{self.description}\n\n" +
                f"Example: {self.example}")

    def __repr__(self):
        return f"{self.name}"

class BaseEvent:
    """ 
    Base class for event monitoring.
    Events structure: dict with guild.id key for each guild the bot is in,
    which each leads to a dict with member.id key for each member in that guild,
    which each leads to a BaseEvent object.
    """
    raw_guilds = get_raw_guilds()
    guild_ids = [int(raw_guild["id"]) for raw_guild in raw_guilds]
    events = {guild_id: {} for guild_id in guild_ids}

    def __init__(self, member, seconds):
        self.member = member
        self.seconds = seconds
        self.start_time = loop.time()

    def get_remaining_seconds(self):
        """ Calculate amount of remaining seconds to end mute. """
        seconds_passed = int(loop.time() - self.start_time)
        return self.seconds - seconds_passed

    @classmethod
    def add_guild(cls, guild_id):
        """ Add guild to event dict. """
        cls.events[guild_id] = {}

    @classmethod
    def remove_guild(cls, guild_id):
        """ Remove guild from event dict. """
        del cls.events[guild_id]

    @classmethod
    def add_event(cls, guild_id, event):
        """ Add event to event guild dict. """
        cls.events[guild_id][event.member.id] = event

    @classmethod
    def remove_event(cls, guild_id, member_id):
        """ Remove event from event guild dict. """
        del cls.events[guild_id][member_id]

    @classmethod
    def verify_guild(cls, guild_id):
        """ Verify if events dict has guild by id. """
        return guild_id in cls.events

    @classmethod
    def verify_guild_member(cls, guild_id, member_id):
        """ Verify if event guild dict has member by id. """
        return member_id in cls.events[guild_id]

    @classmethod
    def get_events(cls, guild_id):
        """ Get list of events in a given guild. """
        return cls.events[guild_id].values()

    @classmethod
    def get_formatted_list(cls, guild_id):
        """ Get formatted list of running mute events in a guild. """
        events = sorted(cls.get_events(guild_id), key=lambda event: event.member.name)
        if len(events) == 0:
            return "There are no muted users"
        
        formatted_events = [f"{event.member.name}: {event.get_remaining_seconds()} seconds" 
                            for event in events]

        return "\n".join(formatted_events)

class MuteEvent(BaseEvent):
    """ Mute event on progress. """
    events = BaseEvent.events.copy()

    def __init__(self, guild, member, seconds):
        super().__init__(member, seconds)
        MuteEvent.add_event(guild.id, self)

class DeafEvent(BaseEvent):
    """ Deaf event on progress. """
    events = BaseEvent.events.copy()

    def __init__(self, guild, member, seconds):
        super().__init__(member, seconds)
        DeafEvent.add_event(guild.id, self)

# ---------- Internal functions ---------- #

def extract_id(s):
    """ Extract id from given string. """
    return int(re.search("\d+", s)[0])

def get_random_pediu():
    """ Get string "pediu" with randomized changes. """
    s = "pediu"

    text = "".join([random.choice([char.lower(), char.upper()]) for char in s])
    spaces = " " * random.randint(1, 3)
    extra = "".join("k" if random.random() < 0.7 else "j" for i in range(random.randint(0, 4)))
    exclamations = "?" * random.randint(0, 3)
    result = text + spaces + extra + exclamations

    return result

def request_images(query_param, max_page=1):
    """ Return image srcs found until a given page from query service. """
    base_url = "http://results.dogpile.com"
    headers = {
        "Accept": "text/html", 
        "User-Agent": "Chrome"
    }

    sc_param = "qf-DXuEaaiozRPb6QfKmikc09MxD9e844AiQ5jU-hXvRzfp6h-X8HwPaNABzBYC5N5IFg0kA9LdCG19Lz1s48kdw3VrlTSiLxyOf_JxyotjDoeO6O2et33UaLXwbd6xmjiUpVL5vrsFPJjDSMUuSju3J4dkCbIVKpyysWXTArbm1J3tqNx1NIJx5y31CXkbJUncvadNNjxxoTaa2wfFyiXCkjBk9fu4yEV9uPjAJOIOyRyDJjJ2GQCYJGkD3YOc75TvjR_BzDFcxbhAGZHgMN70Mxy1rmtISH7mUpWVoPVmcR3rAs9Z0_8pnLoDjFy4USYazFuCGfvGdSTdEtwc-yNlmnRnZU1kDqtRXZaRytsWaOlTAuJtGO6KuA2iZT_eO1-7JiJQP89QDP5aNaBjWiPIpx5r5efEafblGkOZsDdf428n9tbkHmi0lpvmOx1ONE9ndc8sjYKrbRXJa4Kun0caH6XmopvmXi61Dcy-Ia3sNglVz6AmK1ijGtta1zi5YoimxW2OMDm6yrFv2isGFwYCQN_CIGRckbapmwGPIB8r9hybmyH2stzEoaGL4bkcEXgZ8jnNvyxB-aufdQ2JfwiJAWxLYNAgIg2RD-RlIuEV6Pbv02qwW66gKw0qYYxFelwh6JRPdb8TvUvyIxQatH5UrZBxN9nnC497IflzwUvQP2qRDo3xB1rA3RXNgROPo90Y5g0pfI_obMD6W8kZphR9K1Zn_Y34zP0b675J_HcGGYZodLnKtNvF3iqjeZ6-mc2cd5X4I3sD7gmrFozSEGC_x45gT5CedL0ddaIBSDYjm3PE1KQscEdKW1UmJWE_bazAotLdsEMwI6TiMVABFkJ7esNQApRSufUkuPqGbkTC857KuXAnII-XCW8StfTblW8Zk6Op0qOdwNatfOqxERRBAXcjHsbBxNhoBbANV7gaO1JfVC2_DzZgWZwoX9f3iIifoiEj2278sHWwO9XZ2Hb_eTP8jmWRXrItheWgJqQThR341lSb05EAvWIKnwGYLy-Bk1ptPcdASms_y-cL53sbOesQYIrM8TBihp9E-wrZ0a4L0ulU9qifWULWOPIOnO06fcqKsA3Vfr_kyPtJJG2SxGQ5qwuhQW9BX6__K9f2mzGx6zKvqOhBMD2GhkHHruKTbcvcikpGfixfE2vwgSPrpnnDvnSR66VAUbytFcvHqPDnyLb3mathdCDqvcjzHcgLopdQXRPJjJzO9BWlAqFyRav2uauF-XzoWAFFROc9hOgshbzTXe4eulrRQ-nm8Uuim6-rIkkEQrlfvhv6uRymj3Dtx7MlFE6kHsGOyheXqfuFsbMoviGv4IiljafVZdyGuPdOdj-A5FNLht-JCF_aOuHhcXnDnoY4qzuDzxBHJZyxqKrrENx3CALTDP8H_VsKd34u8l6k40u3iKzfO9O0jSjv4z-qksXTMzbjgXWoEnpPvsegvHMHNkhMpVuki2T0VTDECxprw6K3ME_5K8UNXjNQ2xIpvtpGFUoCANpvORXhl-cp22rRdMF4f4fFnXn-Pegkf75lFH1XenFnb_zpgzMfge8tZx5p5nQeYuaB92BdzZy8wYTETS83GkUtpPUFLNk8ilnt2_9EOE8Ulp691AgX8ntsAHcVie0e6J_kNO-_2IHB0bidSsN7JteNIcHVJE9bc-sM4ugGxxz1b-datpgS_OGEzGNTa_qZZh-XufjtTxZfGpTKehky1GxN3Da8fJwmhBhSY9qzeSZo5xQFxo_D6JKkwP7Ko6jkB0aaZJL-I-8c0_gah-Y0F"
    urls = [f"{base_url}/serp?qc=images&q={query_param}&page={page}&sc={sc_param}"
            for page in range(1, max_page + 1)]

    srcs = []
    for url in urls:
        response = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(response.content, features="lxml")
        imgs = soup.select("div > a > img")
        page_srcs = [img["src"] for img in imgs]

        for src in page_srcs:
            srcs.append(src)

    return srcs

def request_tpose_srcs(max_page=1):
    """ Request tpose image srcs. """
    srcs = request_images("tpose", max_page)

    extra_srcs = ["https://cdn.discordapp.com/attachments/596741048525389844/651112131592192098/emeraldo.png"]
    for src in extra_srcs:
        srcs.append(src)

    return srcs

async def process_message(message):
    """ Handle message reply. """
    reply = None
    sent_by_bot = message.author.bot
    is_empty = message.content == ""

    if not sent_by_bot and not is_empty:

        print(f"Message received: {message.content}")
        # Handle special messages
        if message.content.lower().startswith("quem"):
            return get_random_pediu()

        # Verify if message was sent by bot
        bot_was_highlighted = re.search(f"<@\!?{bot_id}>", message.content) != None

        # Handle input
        parameters = re.findall("\S+", message.content)
        command = parameters[0]
        global commands
        command_exists = command in commands_map

        # Reply highlight
        if bot_was_highlighted:
            reply = suggest_help()

        # Reply to help
        elif command == f"{prefix}help":
            reply = get_help(parameters)

        # Run command
        elif command_exists:
            command_function = commands_map[command]
            reply = await command_function(message, parameters)

    return reply

# ---------- Commands ---------- #

def suggest_help():
    """ Simple introduction to be performed when bot is mentioned. """
    return f"Type {prefix}help for more info"

def get_help(parameters):
    """ Provide brief description of all commands or
        extended description for a given command. """
    length = len(parameters)
    if length > 2:
        return "Error: Too many parameters"

    # General help
    if length == 1:
        header = "List of available commands:\n\n"
        commands_data = "\n".join([command.get_brief_data() for command in commands.values()])
        reply = header + commands_data

        return reply
    
    # Specific command help
    try:
        command = parameters[1]
        return commands[command].get_extended_data()
    except KeyError:
        return "Error: Invalid command was given"

async def mute(message, parameters):
    """ Mute someone for given number of seconds. """
    length = len(parameters)
    required_length = 3

    # Validate length
    if length == 1:
        return "Error: No 'member' parameter was given"
    if length == 2:
        return "Error: No 'seconds' parameter was given"
    if length > required_length:
        return "Error: Too many parameters"
    
    member_name = parameters[1]
    member_id = extract_id(member_name)
    seconds_str = parameters[2]

    # Validate author permissions
    author = message.author
    roles = author.roles
    role_can_mute = any([role.permissions.mute_members for role in roles])
    is_admin = any([role.permissions.administrator for role in roles])
    is_owner = author.id == message.guild.owner_id
    can_mute = role_can_mute or is_admin or is_owner

    if not can_mute:
        return f"Error: User '{author.name}' doesn't have permission to mute"

    # Validate member
    members = [member for member in message.guild.members]
    try:
        member = [member for member in members if member.id == member_id][0]
        is_member_valid = True
    except IndexError:
        is_member_valid = False

    if not is_member_valid:
        return f"Error: User '{member_name}' not found on this server"

    # Validate seconds
    try:
        is_seconds_number = re.search("\D", seconds_str) == None
        seconds = int(seconds_str)
        is_seconds_on_bounds = seconds > 0 and seconds <= 3600
        is_seconds_valid = is_seconds_number and is_seconds_on_bounds
    except ValueError:
        is_seconds_valid = False

    if not is_seconds_valid:
        return ("Error: Invalid amount of seconds, " + 
               "it has to be an integer between 1 and 3600")

    # Mute and sleep
    try:
        await member.edit(mute=True)
    except discord.errors.HTTPException as e:
        return "Error: User is not in voice chat"

    mute_event = MuteEvent(message.guild, member, seconds)
    reply = f"User {member.name} is muted for {seconds} seconds"
    await message.channel.send(reply)
    await asyncio.sleep(seconds)
    
    # Unmute if still muted
    guild_id = message.guild.id
    is_muted = MuteEvent.verify_guild_member(guild_id, member.id)
    if is_muted:
        await member.edit(mute=False)

async def unmute(message, parameters):
    """ Unmute a muted member. """
    length = len(parameters)
    required_length = 2

    # Validate length
    if length == 1:
        return "Error: No 'member' parameter was given"
    if length > required_length:
        return "Error: Too many parameters"

    member_name = parameters[1]
    member_id = extract_id(member_name)

    # Validate member
    members = [member for member in message.guild.members]
    try:
        member = [member for member in members if member.id == member_id][0]
        is_member_valid = True
    except IndexError:
        is_member_valid = False

    if not is_member_valid:
        return f"Error: User '{member_name}' not found on this server"

    # Unmute if still muted
    guild_id = message.guild.id
    is_muted = MuteEvent.verify_guild_member(guild_id, member.id)
    if is_muted:
        await member.edit(mute=False)
    else:
        return f"User {member_name} is not time-muted"

async def mutelist(message, parameters):
    """ Get list of currently muted members on requested guild. """
    length = len(parameters)
    required_length = 1

    # Validate length
    if length > required_length:
        return "Error: Too many parameters"

    return MuteEvent.get_formatted_list(message.guild.id)

async def deaf(message, parameters):
    """ Deafen someone for given number of seconds. """
    length = len(parameters)
    required_length = 3

    # Validate length
    if length == 1:
        return "Error: No 'member' parameter was given"
    if length == 2:
        return "Error: No 'seconds' parameter was given"
    if length > required_length:
        return "Error: Too many parameters"
    
    member_name = parameters[1]
    member_id = extract_id(member_name)
    seconds_str = parameters[2]

    # Validate author permissions
    author = message.author
    roles = author.roles
    role_can_deaf = any([role.permissions.deafen_members for role in roles])
    is_admin = any([role.permissions.administrator for role in roles])
    is_owner = author.id == message.guild.owner_id
    can_deaf = role_can_deaf or is_admin or is_owner

    if not can_deaf:
        return f"Error: User '{author.name}' doesn't have permission to deafen"

    # Validate member
    members = [member for member in message.guild.members]
    try:
        member = [member for member in members if member.id == member_id][0]
        is_member_valid = True
    except IndexError:
        is_member_valid = False

    if not is_member_valid:
        return f"Error: User '{member_name}' not found on this server"

    # Validate seconds
    try:
        is_seconds_number = re.search("\D", seconds_str) == None
        seconds = int(seconds_str)
        is_seconds_on_bounds = seconds > 0 and seconds <= 3600
        is_seconds_valid = is_seconds_number and is_seconds_on_bounds
    except ValueError:
        is_seconds_valid = False

    if not is_seconds_valid:
        return ("Error: Invalid amount of seconds, " + 
               "it has to be an integer between 1 and 3600")

    # Mute and sleep
    try:
        await member.edit(deafen=True)
    except discord.errors.HTTPException as e:
        return "Error: User is not in voice chat"

    deaf_event = DeafEvent(message.guild, member, seconds)
    reply = f"User {member.name} is deafened for {seconds} seconds"
    await message.channel.send(reply)
    await asyncio.sleep(seconds)
    
    # Undeaf if still deafened
    guild_id = message.guild.id
    is_deafen = DeafEvent.verify_guild_member(guild_id, member.id)
    if is_deafen:
        await member.edit(deafen=False)

async def undeaf(message, parameters):
    """ Undeaf a deafened member. """
    length = len(parameters)
    required_length = 2

    # Validate length
    if length == 1:
        return "Error: No 'member' parameter was given"
    if length > required_length:
        return "Error: Too many parameters"

    member_name = parameters[1]
    member_id = extract_id(member_name)

    # Validate member
    members = [member for member in message.guild.members]
    try:
        member = [member for member in members if member.id == member_id][0]
        is_member_valid = True
    except IndexError:
        is_member_valid = False

    if not is_member_valid:
        return f"Error: User '{member_name}' not found on this server"

    # Undeaf if still deafened
    guild_id = message.guild.id
    is_deafen = DeafEvent.verify_guild_member(guild_id, member.id)
    if is_deafen:
        await member.edit(deafen=False)
    else:
        return f"User {member_name} is not time-deafened"

async def deaflist(message, parameters):
    """ Get list of currently deafened members on requested guild. """
    length = len(parameters)
    required_length = 1

    # Validate length
    if length > required_length:
        return "Error: Too many parameters"

    return DeafEvent.get_formatted_list(message.guild.id)

async def tpose(message, parameters):
    """ Request random tpose image. """
    length = len(parameters)
    required_length = 1

    # Validate length
    if length > required_length:
        return "Error: Too many parameters"

    global srcs
    src = random.choice(srcs)

    return src

# ---------- Application variables ---------- #

# Async scheduler
loop = asyncio.get_event_loop()

# Tpose image srcs
srcs = request_tpose_srcs(max_page=5)

# Map string to object
commands = {
    "help": Command(f"{prefix}help",
                    ("Provides brief description for all commands, " +
                    "or extended description for a specific command " +
                    "if any is given"),
                    f"Get help for a command\n{prefix}help mute"),
    "mute": Command(f"{prefix}mute",
                    "Mute user for a given number of seconds", 
                    f"Mute Fred for 30 seconds\n{prefix}mute @Fred 30"),
    "unmute": Command(f"{prefix}unmute",
                    "Unmute a muted user", 
                    f"Unmute Fred\n{prefix}unmute @Fred"),
    "mutelist": Command(f"{prefix}mutelist",
                     "Get list of currently muted users in this server",
                     f"\n{prefix}mutelist"),
    "deaf": Command(f"{prefix}deaf",
                    "Deafen user for a given number of seconds", 
                    f"Deafen Fred for 30 seconds\n{prefix}deaf @Fred 30"),
    "undeaf": Command(f"{prefix}undeaf",
                    "Undeaf a deafened user", 
                    f"Undeaf Fred\n{prefix}undeaf @Fred"),
    "deaflist": Command(f"{prefix}deaflist",
                     "Get list of currently deafened users in this server",
                     f"\n{prefix}deaflist"),
    "tpose": Command(f"{prefix}tpose",
                     "Get random tpose image",
                     f"\n{prefix}tpose")
}

# Map string to function
commands_map = {
    f"{prefix}help": get_help,
    f"{prefix}mute": mute,
    f"{prefix}unmute": unmute,
    f"{prefix}mutelist": mutelist,
    f"{prefix}deaf": deaf,
    f"{prefix}undeaf": undeaf,
    f"{prefix}deaflist": deaflist,
    f"{prefix}tpose": tpose
}

bot_id = 647954736959717416
client = discord.Client()

# ---------- Event listeners ---------- #

# Bot start
@client.event
async def on_ready():
    print(f"{client.user} awoke")

# Send message
@client.event
async def on_message(message):
    try:
        reply = await process_message(message)
        if reply != None:
            await message.channel.send(reply)
    except UnicodeEncodeError:
        pass

# Join, leave, mute, deafen on VC
@client.event
async def on_voice_state_update(member, before, after):
    was_unmuted = before.mute and not after.mute
    was_undeafen = before.deaf and not after.deaf
    guild_id = member.guild.id

    # Remove dict element on unmute
    if was_unmuted:
        if MuteEvent.verify_guild_member(guild_id, member.id):
            MuteEvent.remove_event(guild_id, member.id)

    # Remove dict element on undeaf
    if was_undeafen:
        if DeafEvent.verify_guild_member(guild_id, member.id):
            DeafEvent.remove_event(guild_id, member.id)

# Bot join guild
@client.event
async def on_guild_join(guild):
    was_bot_added = not MuteEvent.verify_guild(guild.id)

    # Add guild dict
    if was_bot_added:
        MuteEvent.add_guild(guild.id)
        DeafEvent.add_guild(guild.id)

# Member leave guild
@client.event
async def on_member_remove(member):
    was_bot_removed = member.id == bot_id

    # Remove guild dict
    if was_bot_removed:
        MuteEvent.remove_guild(member.guild.id)
        DeafEvent.remove_guild(member.guild.id)

client.run(token)
