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
base_path = "C:/Users/Usuario/Downloads/discord-bot"
os.chdir(base_path)

env_path = "./.env"
bad_words_path = "./bad-words.txt"
extra_srcs_path = "./extra-srcs.txt"

dotenv.load_dotenv(env_path)

# ---------- Environment variables ---------- #

prefix = os.getenv("prefix")
token = os.getenv("token")
sc_param = os.getenv("sc_param")

with open(bad_words_path) as file:
    data = file.read()
bad_word_regexes = json.loads(data)

with open(extra_srcs_path) as file:
    data = file.read()
extra_srcs = json.loads(data)

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
        return self.name

class Restriction:
    """ Restriction filled with data to be used as restrict argument. """
    def __init__(self, name_present, name_past, function_role, restrict_event_class, kwarg):
        self.name_present = name_present
        self.name_past = name_past
        self.function_role = function_role
        self.restrict_event_class = restrict_event_class
        self.kwarg_enable = {kwarg: True}
        self.kwarg_disable = {kwarg: False}

    def __repr__(self):
        return self.name_present

class RestrictEvent:
    """
    Base class for event monitoring.
    Events structure: dict with guild.id key for each guild the bot is in,
    which each leads to a dict with member.id key for each member in that guild,
    which each leads to a RestrictEvent object.
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
            return "List is empty"
        
        formatted_events = [f"{event.member.name}: {event.get_remaining_seconds()} seconds" 
                            for event in events]

        return "\n".join(formatted_events)

class MuteEvent(RestrictEvent):
    """ Mute event on progress. """
    events = RestrictEvent.events.copy()

    def __init__(self, guild, member, seconds):
        super().__init__(member, seconds)
        MuteEvent.add_event(guild.id, self)

class DeafEvent(RestrictEvent):
    """ Deaf event on progress. """
    events = RestrictEvent.events.copy()

    def __init__(self, guild, member, seconds):
        super().__init__(member, seconds)
        DeafEvent.add_event(guild.id, self)

class AmputateEvent(RestrictEvent):
    """ Deaf event on progress. """
    events = RestrictEvent.events.copy()

    def __init__(self, guild, member, seconds):
        super().__init__(member, seconds)
        AmputateEvent.add_event(guild.id, self)

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

    for src in extra_srcs:
        srcs.append(src)

    return srcs

def replace_bad_words(s):
    """ Replace bad words in a message. """
    words = re.findall("\S+", s) or []

    bad_words = {word for word in words 
                 if any(word.search(bad_word_regex, word.lower()) != None) 
                 for bad_word_regex in bad_word_regexes}
    
    for bad_word in bad_words:
        s = re.sub(bad_word, "#" * len(bad_word), s)

    return s

async def restrict(message, parameters, restriction):
    """ Restrict someone for given number of seconds. """
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
    function_role = restriction.function_role

    author = message.author
    roles = author.roles
    role_has_permission = any([function_role(role) for role in roles])
    is_admin = any([role.permissions.administrator for role in roles])
    is_owner = author.id == message.guild.owner_id
    has_permission = role_has_permission or is_admin or is_owner

    if not has_permission:
        return f"Error: User '{author.name}' doesn't have permission to {restriction.name_present}"

    # Validate member
    members = [member for member in message.guild.members]
    member = next((member for member in members if member.id == member_id), None)
    is_member_valid = member != None

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

    # Restrict and sleep
    try:
        kwarg = restriction.kwarg_enable
        await member.edit(**kwarg)
    except discord.errors.HTTPException as e:
        voice_restrictions = {"deaf", "mute"}
        if restriction.name_present in voice_restrictions:
            return "Error: User is not in voice chat"
        else:
            return "Error: User is not in text chat"

    restrict_event_class = restriction.restrict_event_class

    mute_event = restrict_event_class(message.guild, member, seconds)
    reply = f"User {member.name} is {restriction.name_past} for {seconds} seconds"
    await message.channel.send(reply)
    await asyncio.sleep(seconds)
    
    # Unrestrict if still restricted
    guild_id = message.guild.id
    is_restricted = restrict_event_class.verify_guild_member(guild_id, member.id)
    if is_restricted:
        kwarg = restriction.kwarg_disable
        await member.edit(**kwarg)

async def unrestrict(message, parameters, restriction):
    """ Unrestrict a restricted member. """
    length = len(parameters)
    required_length = 2

    # Validate length
    if length == 1:
        return "Error: No 'member' parameter was given"
    if length > required_length:
        return "Error: Too many parameters"

    member_name = parameters[1]
    member_id = extract_id(member_name)

    # Validate author permissions
    function_role = restriction.function_role

    author = message.author
    roles = author.roles
    role_has_permission = any([function_role(role) for role in roles])
    is_admin = any([role.permissions.administrator for role in roles])
    is_owner = author.id == message.guild.owner_id
    has_permission = role_has_permission or is_admin or is_owner

    if not has_permission:
        return f"Error: User '{author.name}' doesn't have permission to un{restriction.name_present}"

    # Validate member
    members = [member for member in message.guild.members]
    member = next((member for member in members if member.id == member_id), None)
    is_member_valid = member != None

    if not is_member_valid:
        return f"Error: User '{member_name}' not found on this server"

    restrict_event_class = restriction.restrict_event_class

    # Unrestrict if still restricted
    guild_id = message.guild.id
    is_restricted = restrict_event_class.verify_guild_member(guild_id, member.id)
    if is_restricted:
        kwarg = restriction.kwarg_disable
        await member.edit(**kwarg)
    else:
        return f"User {member_name} is not time-{restriction.name_past}"

async def restrictionlist(message, parameters, restriction):
    """ Get list of currently restricted members on requested guild. """
    length = len(parameters)
    required_length = 1

    # Validate length
    if length > required_length:
        return "Error: Too many parameters"

    restrict_event_class = restriction.restrict_event_class

    return restrict_event_class.get_formatted_list(message.guild.id)

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

        # Verify if message is just bot highlight
        bot_was_highlighted = re.search(f"^<@\!?{bot_id}>$", message.content) != None

        # Handle input
        parameters = re.findall("\S+", message.content)
        command = parameters[0]
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
    """ Simple introduction to be performed when bot is highlighted. """
    return f"Type {prefix}help for more info"

def get_help(parameters):
    """ Describe all commands briefly or a given command extensively. """
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
    restriction = restriction_map["mute"]
    return await restrict(message, parameters, restriction)

async def unmute(message, parameters):
    """ Unmute a muted member. """
    restriction = restriction_map["mute"]
    return await unrestrict(message, parameters, restriction)

async def mutelist(message, parameters):
    """ Get list of currently muted members on requested guild. """
    restriction = restriction_map["mute"]
    return await restrictionlist(message, parameters, restriction)

async def deaf(message, parameters):
    """ Deafen someone for given number of seconds. """
    restriction = restriction_map["deaf"]
    return await restrict(message, parameters, restriction)

async def undeaf(message, parameters):
    """ Undeaf a deafened member. """
    restriction = restriction_map["deaf"]
    return await unrestrict(message, parameters, restriction)

async def deaflist(message, parameters):
    """ Get list of currently deafened members on requested guild. """
    restriction = restriction_map["deaf"]
    return await restrictionlist(message, parameters, restriction)

async def amputate(message, parameters):
    """ Amputate someone for given number of seconds. """
    restriction = restriction_map["amputate"]
    return await restrict(message, parameters, restriction)

async def unamputate(message, parameters):
    """ Unamputate a amputated member. """
    restriction = restriction_map["amputate"]
    return await unrestrict(message, parameters, restriction)

async def deaflist(message, parameters):
    """ Get list of currently amputated members on requested guild. """
    restriction = restriction_map["amputate"]
    return await restrictionlist(message, parameters, restriction)

async def serverlist(message, parameters):
    """ Request list of servers in which TPoseBot is present. """
    length = len(parameters)
    required_length = 1

    # Validate length
    if length > required_length:
        return "Error: Too many parameters"

    guilds = client.guilds
    guild_names = [guild.name for guild in guilds]

    guilds = sorted(guilds, key=lambda guild: guild.name)
    formatted_guilds = [f"{guild.name} | Member count: {len(guild.members)}" 
                       for guild in guilds]

    return '\n'.join(formatted_guilds) + "\n\n" + f"Amount: {len(formatted_guilds)}"

async def tpose(message, parameters):
    """ Request random tpose image. """
    length = len(parameters)
    required_length = 1

    # Validate length
    if length > required_length:
        return "Error: Too many parameters"

    src = random.choice(srcs)

    return src

# ---------- Application variables ---------- #

# Async scheduler
loop = asyncio.get_event_loop()

# Tpose image srcs
srcs = request_tpose_srcs(max_page=5)

# Map string to command description object
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
    "serverlist": Command(f"{prefix}serverlist",
                     "Get list of servers in which TPoseBot is present",
                     f"\n{prefix}serverlist"),
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
    f"{prefix}serverlist": serverlist,
    f"{prefix}tpose": tpose
}

# Map string to restriction data object
restriction_map = {
    "amputate": Restriction("amputate", 
                            "amputated", 
                            lambda role: role.permissions.manage_messages, 
                            AmputateEvent, 
                            "roles"),
    "deaf": Restriction("deaf",
                        "deafen", 
                        lambda role: role.permissions.deafen_members, 
                        DeafEvent,
                        "deaf"),
    "mute": Restriction("mute", 
                        "muted", 
                        lambda role: role.permissions.mute_members, 
                        MuteEvent,
                        "mute")
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
