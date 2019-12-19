# Safe class annotations
from __future__ import annotations
# General types
from typing import *

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
with open("C:/Users/fredt/Downloads/discord-bot/bot.py") as file:
    exec(file.read())
"""

# ---------- Environment variables ---------- #

base_path: str = "C:/Users/fredt/Downloads/discord-bot"
env_path: str = "./.env"
os.chdir(base_path)
dotenv.load_dotenv(env_path)

prefix: str = os.getenv("prefix")
token: str = os.getenv("token")
sc_param: str = os.getenv("sc_param")

bad_words_path: str = "./bad-words.txt"
extra_srcs_path: str = "./extra-srcs.txt"
report_path: str = "E:/tposebot-data/bot-reports.txt"

# ---------- Load environment ---------- #

with open(bad_words_path) as file:
    data: str = file.read()
    bad_word_regexes: List[str] = json.loads(data)

with open(extra_srcs_path) as file:
    data: str = file.read()
    extra_srcs: List[str] = json.loads(data)

# ---------- Class helpers ---------- #

def get_raw_guilds() -> List[Dict]:
    """ Get guilds in which the bot exists. """
    url: str = "https://discordapp.com/api/v6/users/@me/guilds"
    headers: Dict = {"Authorization": f"Bot {token}"}
    response: requests.Response = requests.get(url, headers=headers)
    raw_guilds: List[Dict] = json.loads(response.content)

    return raw_guilds

# ---------- Classes ---------- #

class Command:
    """ Describe a command available from bot. """
    def __init__(self, name: str, description: str, example: str):
        self.name = name
        self.description = description
        self.example = example

    def get_brief_data(self) -> str:
        return f"{self.name}: {self.description}"

    def get_extended_data(self) -> str:
        return (f"Description: \n{self.description}\n\n" +
                f"Example: {self.example}")

    def __repr__(self):
        return self.name

class RestrictEvent:
    """ Base class for event monitoring. """
    raw_guilds: List[Dict] = get_raw_guilds()
    guild_ids: List[int] = [int(raw_guild["id"]) for raw_guild in raw_guilds]

    events: Dict[int, Dict[int, RestrictEvent]] = {guild_id: {} 
                                                   for guild_id in guild_ids}

    def __init__(self, member: discord.Member, seconds: int):
        self.member: discord.Member = member
        self.seconds: int = seconds
        self.start_time: float = loop.time()

    def get_remaining_seconds(self) -> int:
        """ Calculate amount of remaining seconds to end mute. """
        seconds_passed: int = int(loop.time() - self.start_time)
        remaining_seconds: int = self.seconds - seconds_passed
        return remaining_seconds

    @classmethod
    def add_guild(cls, guild_id: int) -> None:
        """ Add guild to event dict. """
        cls.events[guild_id] = {}

    @classmethod
    def remove_guild(cls, guild_id: int) -> None:
        """ Remove guild from event dict. """
        del cls.events[guild_id]

    @classmethod
    def add_event(cls, guild_id: int, event: RestrictEvent) -> None:
        """ Add event to event guild dict. """
        cls.events[guild_id][event.member.id] = event

    @classmethod
    def remove_event(cls, guild_id: int, member_id: int) -> None:
        """ Remove event from event guild dict. """
        del cls.events[guild_id][member_id]

    @classmethod
    def verify_guild(cls, guild_id: int) -> bool:
        """ Verify if events dict has guild by id. """
        return guild_id in cls.events

    @classmethod
    def verify_guild_member(cls, guild_id: int, member_id: int) -> bool:
        """ Verify if event guild dict has member by id. """
        return member_id in cls.events[guild_id]

    @classmethod
    def get_events(cls, guild_id: int) -> Iterator[RestrictEvent]:
        """ Get list of events in a given guild. """
        return cls.events[guild_id].values()

    @classmethod
    def get_formatted_list(cls, guild_id: int) -> str:
        """ Get formatted list of running mute events in a guild. """
        events: List[RestrictEvent] = sorted(cls.get_events(guild_id), 
                                             key=lambda event: event.member.name)
        if len(events) == 0:
            return "List is empty"
        
        formatted_events: str = [(f"{event.member.name}: " + 
                                  f"{event.get_remaining_seconds()} seconds") 
                                  for event in events]

        return "\n".join(formatted_events)

class MuteEvent(RestrictEvent):
    """ Mute event on progress. """
    events: Dict[int, Dict[int, RestrictEvent]] = RestrictEvent.events.copy()

    def __init__(self, guild: discord.Guild, member: discord.Member, seconds: int):
        super().__init__(member, seconds)
        MuteEvent.add_event(guild.id, self)

class DeafEvent(RestrictEvent):
    """ Deaf event on progress. """
    events: Dict[int, Dict[int, RestrictEvent]] = RestrictEvent.events.copy()

    def __init__(self, guild: discord.Guild, member: discord.Member, seconds: int):
        super().__init__(member, seconds)
        DeafEvent.add_event(guild.id, self)

class AmputateEvent(RestrictEvent):
    """ Amputate event on progress. """
    events: Dict[int, Dict[int, RestrictEvent]] = RestrictEvent.events.copy()

    def __init__(self, guild: discord.Guild, member: discord.Member, seconds: int):
        super().__init__(member, seconds)
        AmputateEvent.add_event(guild.id, self)

class Restriction:
    """ Restriction filled with data to be used as restrict argument. """
    def __init__(self, name_present: str, 
                       name_past: str, 
                       function_role: Callable, 
                       restrict_event_class: ClassVar[RestrictEvent]):
        self.name_present: str = name_present
        self.name_past: str = name_past
        self.function_role: Callable = function_role
        self.restrict_event_class: ClassVar[RestrictEvent] = restrict_event_class

    def __repr__(self):
        return self.name_present

class BooleanRestriction(Restriction):
    """ Restriction managed through boolean value. """
    def __init__(self, name_present: str, 
                       name_past: str, 
                       function_role: Callable, 
                       restrict_event_class: ClassVar[RestrictEvent], 
                       kwarg_key: str):
        super().__init__(name_present, name_past, function_role, restrict_event_class)
        self.kwarg_enable: Dict[str, bool] = {kwarg_key: True}
        self.kwarg_disable: Dict[str, bool] = {kwarg_key: False}

class RoleRestriction(Restriction):
    """ Restriction managed through role list. """
    def __init__(self, name_present: str, 
                       name_past: str, 
                       function_role: Callable, 
                       restrict_event_class: ClassVar[RestrictEvent], 
                       kwarg_key: str):
        super().__init__(name_present, name_past, function_role, restrict_event_class)
        self.kwarg_enable = "nada"
        self.kwarg_disable = "nada"

# ---------- Internal functions ---------- #

def extract_id(s: str) -> int:
    """ Extract id from given string. """
    return int(re.search("\d+", s)[0])

def get_random_pediu() -> str:
    """ Get string "pediu" with randomized changes. """
    s: str = "pediu"

    text: str = "".join([random.choice([char.lower(), char.upper()]) for char in s])
    spaces: str = " " * random.randint(1, 3)
    laughs: str = "".join("k" if random.random() < 0.7 else "j" for i in range(random.randint(0, 4)))
    exclamations: str = "?" * random.randint(0, 3)
    result: str = text + spaces + laughs + exclamations

    return result

def request_image_srcs(query_param: str, max_page: int = 1) -> List[str]:
    """ Return image srcs found until a given page from query service. """
    base_url: str = "http://results.dogpile.com"
    headers: Dict = {
        "Accept": "text/html", 
        "User-Agent": "Chrome"
    }

    urls: List[str] = [f"{base_url}/serp?qc=images&q={query_param}&page={page}&sc={sc_param}"
                       for page in range(1, max_page + 1)]

    srcs: List[str] = []
    for url in urls:
        response: requests.Response = requests.get(url, headers=headers)
        soup: bs4.BeautifulSoup = bs4.BeautifulSoup(response.content, features="lxml")
        imgs: bs4.element.Tag = soup.select("div > a > img")
        page_srcs: List[str] = [img["src"] for img in imgs]
        srcs.extend(page_srcs)

    return srcs

def request_tpose_srcs(max_page: int = 1) -> List[str]:
    """ Request tpose image srcs. """
    srcs: List[str] = request_image_srcs("tpose", max_page)
    srcs.extend(extra_srcs)

    return srcs

def replace_bad_words(s: str) -> Tuple[str, int]:
    """ Replace bad words in a message. Return filtered message and amount of filtered words. """
    words: List[str] = re.findall("\S+", s) or []

    bad_words: Set[str] = {word for word in words 
                           if any(re.search(f"\\b{bad_word_regex}\\b", word.lower()) != None 
                           for bad_word_regex in bad_word_regexes)}
    
    for bad_word in bad_words:
        s = s.replace(bad_word, "#" * len(bad_word))

    bad_word_amount: int = sum([word == bad_word for word in words for bad_word in bad_words])

    return s, bad_word_amount

async def restrict(message: discord.Message, parameters: List[str], 
                                             restriction: RestrictEvent) -> str:
    """ Restrict someone for given number of seconds. """
    length: int = len(parameters)
    required_length: int = 3

    # Validate length
    if length == 1:
        return "No 'member' parameter was given"
    if length == 2:
        return "No 'seconds' parameter was given"
    if length > required_length:
        return "Too many parameters"
    
    member_name: str = parameters[1]
    member_id: int = extract_id(member_name)
    seconds_str: str = parameters[2]

    # Validate author permissions
    function_role: Callable = restriction.function_role

    roles: List[discord.Role] = message.author.roles
    role_has_permission: bool = any([function_role(role) for role in roles])
    is_admin: bool = any([role.permissions.administrator for role in roles])
    is_owner: bool = message.author.id == message.guild.owner_id
    has_permission: bool = role_has_permission or is_admin or is_owner

    if not has_permission:
        return f"User '{message.author.name}' doesn't have permission to {restriction.name_present}"

    # Validate member
    members: List[discord.Member] = [member for member in message.guild.members]
    member: Member = next((member for member in members if member.id == member_id), None)
    is_member_valid: bool = member != None

    if not is_member_valid:
        return f"User '{member_name}' not found on this server"

    # Validate seconds
    try:
        is_seconds_number: bool = re.search("\D", seconds_str) == None
        seconds: int = int(seconds_str)
        min_seconds_amount: int = 1
        max_seconds_amount: int = 7200
        is_seconds_on_bounds: bool = (seconds >= min_seconds_amount and 
                                      seconds <= max_seconds_amount)
        is_seconds_valid: bool = is_seconds_number and is_seconds_on_bounds
    except ValueError:
        is_seconds_valid: bool = False

    if not is_seconds_valid:
        return ("Invalid amount of seconds, it has to be an integer " +
                f"between {min_seconds_amount} and {max_seconds_amount}")

    # Restrict and sleep
    try:
        kwarg: Dict = restriction.kwarg_enable
        await member.edit(**kwarg)
    except discord.errors.HTTPException as e:
        if restriction.name_present in voice_restrictions:
            return "User is not in voice chat"

    restrict_event_class: ClassVar[RestrictEvent] = restriction.restrict_event_class

    mute_event: RestrictEvent = restrict_event_class(message.guild, member, seconds)
    reply: str = f"User {member.name} is {restriction.name_past} for {seconds} seconds"
    await message.channel.send(reply)
    await asyncio.sleep(seconds)

    # Unrestrict if still restricted
    guild_id: int = message.guild.id
    is_restricted: bool = restrict_event_class.verify_guild_member(guild_id, member.id)
    if is_restricted:
        kwarg: Dict = restriction.kwarg_disable
        await member.edit(**kwarg)

async def unrestrict(message: discord.Message, parameters: List[str], 
                                               restriction: RestrictEvent) -> str:
    """ Unrestrict a restricted member. """
    length: int = len(parameters)
    required_length: int = 2

    # Validate length
    if length == 1:
        return "No 'member' parameter was given"
    if length > required_length:
        return "Too many parameters"

    member_name: str = parameters[1]
    member_id: int = extract_id(member_name)

    # Validate author permissions
    function_role: Callable = restriction.function_role

    roles: List[discord.Role] = message.author.roles
    role_has_permission: bool = any([function_role(role) for role in roles])
    is_admin: bool = any([role.permissions.administrator for role in roles])
    is_owner: bool = message.author.id == message.guild.owner_id
    has_permission: bool = role_has_permission or is_admin or is_owner

    if not has_permission:
        return f"User '{message.author.name}' doesn't have permission to {restriction.name_present}"

    # Validate member
    members: List[discord.Member] = [member for member in message.guild.members]
    member: Member = next((member for member in members if member.id == member_id), None)
    is_member_valid: bool = member != None

    if not is_member_valid:
        return f"User '{member_name}' not found on this server"

    restrict_event_class: ClassVar[RestrictEvent] = restriction.restrict_event_class

    # Unrestrict if still restricted
    guild_id: int = message.guild.id
    is_restricted: bool = restrict_event_class.verify_guild_member(guild_id, member.id)
    if is_restricted:
        kwarg: Dict = restriction.kwarg_disable
        await member.edit(**kwarg)
    else:
        return f"User {member_name} is not time-{restriction.name_past}"

async def restrictionlist(message: discord.Message, parameters: List[str], 
                                                    restriction: RestrictEvent) -> str:
    """ Get list of currently restricted members on requested guild. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        return "Too many parameters"

    restrict_event_class: ClassVar[RestrictEvent] = restriction.restrict_event_class

    return restrict_event_class.get_formatted_list(message.guild.id)

async def process_message(message: discord.Message) -> str:
    """ Handle message reply. """
    reply: str = None
    sent_by_bot: bool = message.author.bot
    is_empty: bool = message.content == ""

    if not sent_by_bot and not is_empty:

        print(f"Message received: {message.content}")
        
        replaced_content, bad_word_amount = replace_bad_words(message.content)

        # Handle special messages
        if message.content.lower().startswith("quem"):
            return get_random_pediu()

        # Verify if message is just bot highlight
        bot_was_highlighted: bool = re.search(f"^<@\!?{bot_id}>$", message.content) != None

        # Handle input
        parameters: List[str] = re.findall("\S+", message.content)
        command: str = parameters[0]
        is_command_help: bool = command == f"{prefix}help"
        command_exists: bool = command in commands_map

        # Reply highlight
        if bot_was_highlighted:
            reply: str = suggest_help()

        # Reply to help
        elif is_command_help:
            reply: str = get_help(parameters)

        # Run command
        elif command_exists:
            command_function: Callable = commands_map[command]
            reply: str = await command_function(message, parameters)

    return reply

# ---------- Commands ---------- #

def suggest_help() -> str:
    """ Simple introduction to be performed when bot is highlighted. """
    return f"Type {prefix}help for more info"

def get_help(parameters: List[str]) -> str:
    """ Describe all commands briefly or a given command extensively. """
    length: int = len(parameters)
    required_lengths: Set[int] = {1, 2}
    max_required_length: int = max(required_lengths)
    is_general_help: bool = length == 1
    is_command_help: bool = length == 2

    if length > max_required_length:
        return "Too many parameters"

    # General help
    if is_general_help:
        header: str = "List of available commands:\n\n"
        commands_data: str = [command.get_brief_data() for command in commands.values()]
        formatted_commands_data: str = "\n".join(commands_data)
        reply: str = header + formatted_commands_data

        return reply
    
    # Specific command help
    if is_command_help:
        try:
            command: str = parameters[1]
            return commands[command].get_extended_data()
        except KeyError:
            return "Invalid command was given"

async def report(message: discord.Message, parameters: List[str]) -> str:
    """ Report a bug. """
    length = len(parameters)
    min_required_length = 2

    if length < min_required_length:
        return "No 'message' parameter was given"

    content = message.content
    report_message_match = re.search(f"{prefix}report\s+", content)
    report_message_index = report_message_match.end()
    report_message = content[report_message_index :]

    min_message_length = 5
    is_message_length_low = len(report_message) < min_message_length

    if is_message_length_low:
        return (f"Message length is too low, " +
                f"should have at least {min_message_length} characters")

    with open(report_path, "a") as file:
        file.write(f"- {report_message}\n")
    return "Thank you for helping!"

async def amputate(message: discord.Message, parameters: List[str]) -> str:
    """ Amputate someone for given number of seconds. """
    restriction: RestrictEvent = restriction_map["amputate"]
    return await restrict(message, parameters, restriction)

async def unamputate(message: discord.Message, parameters: List[str]) -> str:
    """ Unamputate a amputated member. """
    restriction: RestrictEvent = restriction_map["amputate"]
    return await unrestrict(message, parameters, restriction)

async def amputatelist(message: discord.Message, parameters: List[str]) -> str:
    """ Get list of currently amputated members on requested guild. """
    restriction: RestrictEvent = restriction_map["amputate"]
    return await restrictionlist(message, parameters, restriction)

async def deaf(message: discord.Message, parameters: List[str]) -> str:
    """ Deafen someone for given number of seconds. """
    restriction: RestrictEvent = restriction_map["deaf"]
    return await restrict(message, parameters, restriction)

async def undeaf(message: discord.Message, parameters: List[str]) -> str:
    """ Undeaf a deafened member. """
    restriction: RestrictEvent = restriction_map["deaf"]
    return await unrestrict(message, parameters, restriction)

async def deaflist(message: discord.Message, parameters: List[str]) -> str:
    """ Get list of currently deafened members on requested guild. """
    restriction: RestrictEvent = restriction_map["deaf"]
    return await restrictionlist(message, parameters, restriction)

async def mute(message: discord.Message, parameters: List[str]) -> str:
    """ Mute someone for given number of seconds. """
    restriction: RestrictEvent = restriction_map["mute"]
    return await restrict(message, parameters, restriction)

async def unmute(message: discord.Message, parameters: List[str]) -> str:
    """ Unmute a muted member. """
    restriction: RestrictEvent = restriction_map["mute"]
    return await unrestrict(message, parameters, restriction)

async def mutelist(message: discord.Message, parameters: List[str]) -> str:
    """ Get list of currently muted members on requested guild. """
    restriction: RestrictEvent = restriction_map["mute"]
    return await restrictionlist(message, parameters, restriction)

async def serverlist(message: discord.Message, parameters: List[str]) -> str:
    """ Request list of servers in which TPoseBot is present. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        return "Too many parameters"

    guilds: List[discord.Guild] = client.guilds
    header: str = f"{len(guilds)} servers found:\n\n"
    guild_names: List[str] = [guild.name for guild in guilds]

    guilds: List[str] = sorted(guilds, key=lambda guild: guild.name)
    guilds_data: str = [f"{guild.name} | Member count: {len(guild.members)}" 
                        for guild in guilds]
    formatted_guilds: str = "\n".join(guilds_data)

    return header + formatted_guilds

async def tpose(message: discord.Message, parameters: List[str]) -> str:
    """ Request random tpose image. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        return "Too many parameters"

    src: str = random.choice(srcs)

    return src

# ---------- Application variables ---------- #

bot_id: str = 647954736959717416
client: discord.Client = discord.Client()

# Async scheduler
loop: asyncio.ProactorEventLoop = asyncio.get_event_loop()

# Tpose image srcs
srcs: List[str] = request_tpose_srcs(max_page=3)

# Map string to command object
commands: Dict[str, Command] = {
    "help": Command(f"{prefix}help",
                    ("Provides brief description for all commands, " +
                    "or extended description for a specific command " +
                    "if any is given"),
                    f"Get help for a command\n{prefix}help mute"),
    "report": Command(f"{prefix}report",
                    "Report a bug", 
                    (f"Report a permission related bug" + 
                    f"\n{prefix}report administrators are not having permission to mute")),
    "amputate": Command(f"{prefix}amputate",
                    ("Amputate user for a given number of seconds, creates role 'amputated'," +
                     "amputated users cannot send messages in text channels"), 
                    f"Amputate Fred for 30 seconds\n{prefix}amputate @Fred 30"),
    "unamputate": Command(f"{prefix}unamputate",
                    "Unamputate a amputated user", 
                    f"Unamputate Fred\n{prefix}unamputate @Fred"),
    "amputatelist": Command(f"{prefix}amputatelist",
                     "Get list of currently amputated users in this server",
                     f"\n{prefix}amputatelist"),
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
commands_map: Dict[str, Callable] = {
    f"{prefix}help": get_help,
    f"{prefix}report": report,
    f"{prefix}amputate": amputate,
    f"{prefix}unamputate": unamputate,
    f"{prefix}amputatelist": amputatelist,
    f"{prefix}deaf": deaf,
    f"{prefix}undeaf": undeaf,
    f"{prefix}deaflist": deaflist,
    f"{prefix}mute": mute,
    f"{prefix}unmute": unmute,
    f"{prefix}mutelist": mutelist,
    f"{prefix}serverlist": serverlist,
    f"{prefix}tpose": tpose
}

# Map string to restriction data object
restriction_map: Dict[str, Restriction] = {
    "amputate": RoleRestriction("amputate", 
                                "amputated", 
                                lambda role: role.permissions.manage_messages, 
                                AmputateEvent, 
                                "roles"),
    "deaf": BooleanRestriction("deaf",
                               "deafened", 
                               lambda role: role.permissions.deafen_members, 
                               DeafEvent,
                               "deafen"),
    "mute": BooleanRestriction("mute", 
                               "muted", 
                               lambda role: role.permissions.mute_members, 
                               MuteEvent,
                               "mute")
}

event_classes: Set[ClassVar[Restriction]] = {AmputateEvent, DeafEvent, MuteEvent}
voice_restrictions: Set[str] = {"deaf", "mute"}

# ---------- Event listeners ---------- #

# Bot start
@client.event
async def on_ready():
    print(f"{client.user} awoke")

# Send message
@client.event
async def on_message(message: discord.Message):
    try:
        reply: str = await process_message(message)
        if reply != None:
            await message.channel.send(reply)
    except UnicodeEncodeError:
        pass

# Join, leave, mute, deafen on VC
@client.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, 
                                                        after: discord.VoiceState):
    was_unmuted: bool = before.mute and not after.mute
    was_undeafen: bool = before.deaf and not after.deaf
    guild_id: int = member.guild.id

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
async def on_guild_join(guild: discord.Guild):
    was_bot_added: bool = not MuteEvent.verify_guild(guild.id)

    # Add guild dict
    if was_bot_added:
        for event_class in event_classes:
            event_class.add_guild(guild.id)

# Member leave guild
@client.event
async def on_member_remove(member: discord.Member):
    was_bot_removed: bool = member.id == bot_id

    # Remove guild dict
    if was_bot_removed:
        for event_class in event_classes:
            event_class.remove_guild(member.guild.id)

client.run(token)
