# Safe class annotations
from __future__ import annotations
# General types
from typing import *
# Abstract classes and methods
from abc import ABC, abstractmethod

import aiohttp
import asyncio
import bs4
import copy
import datetime
import discord
import dotenv
import json
import librosa
import os
import random
import re
import requests
import time
import youtube_dl

# ---------- Environment variables ---------- #

base_path: str = "E:/tposebot"
env_path: str = "./.env"
os.chdir(base_path)
dotenv.load_dotenv(env_path)

api_key: str = os.getenv("api_key")
prefix: str = os.getenv("prefix")
token: str = os.getenv("token")
sc_param: str = os.getenv("sc_param")

bad_words_path: str = "./bad-words.txt"
extra_srcs_path: str = "./extra-srcs.txt"
report_path: str = "./reports.txt"

youtube_videos_source_dir: str = "youtube-videos-source"
cursed_audios_dir: str = "cursed-audios"
youtube_videos_download_dir: str = "youtube-videos-download"

n_words_path: str = "./n-words.txt"

with open(n_words_path) as file:
    n_words: List[str] = file.readlines()

dirnames: List[str] = [
    youtube_videos_source_dir,
    cursed_audios_dir,
    youtube_videos_download_dir
]

for dirname in dirnames:
    file_exists: bool = os.path.exists(f"./{dirname}")
    if not file_exists:
        os.mkdir(dirname)


# ---------- Text formatting ---------- #


def format_bold_text(text: str) -> str:
    """ Format text to become bold. """
    return f"**{text}**"


def format_code_block(text: str) -> str:
    """ Format text to become multiline code block. """
    return f"```{text}```"


# ---------- Load environment ---------- #

with open(bad_words_path) as file:
    data: str = file.read()
    bad_word_regexes: List[str] = json.loads(data)

with open(extra_srcs_path) as file:
    data: str = file.read()
    extra_srcs: List[str] = json.loads(data)

with open(report_path) as file:
    data: str = file.read()
    try:
        reports: Set[str] = set(json.loads(data))
    except json.JSONDecodeError:
        reports: Set[str] = set()


# ---------- Class helpers ---------- #


def suggest_help() -> str:
    """ Simple introduction to be performed when bot is highlighted. """
    return f"Type {prefix}help for more info"


bot_id: int = 647954736959717416
game: discord.Game = discord.Game(suggest_help())
client: discord.Client = discord.Client(activity=game)


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


class RestrictEvent(ABC):
    """ Base class for member restriction event management. """
    ids: Dict[int, Dict[int, int]] = None
    events: Dict[int, Dict[int, RestrictEvent]] = None

    def __init__(self, member: discord.Member, seconds: int):
        self.id = self.__class__.ids[member.guild.id]
        self.member: discord.Member = member
        self.seconds: int = seconds
        self.start_time: float = time.time()

        self.__class__.ids[member.guild.id] += 1

    def get_remaining_seconds(self) -> int:
        """ Calculate amount of remaining seconds to end mute. """
        seconds_passed: int = int(time.time() - self.start_time)
        remaining_seconds: int = self.seconds - seconds_passed
        return remaining_seconds

    def get_recent_state(self) -> bool:
        """ Inform if event just happened. """
        return time.time() - self.start_time < 0.5

    @property
    def name_present(self):
        raise NotImplementedError("RestrictEvent.name_present not implemented")

    @property
    def name_past(self):
        raise NotImplementedError("RestrictEvent.name_past not implemented")

    @property
    def function_role(self):
        raise NotImplementedError("RestrictEvent.function_role not implemented")

    @property
    def kwarg_key(self):
        raise NotImplementedError("RestrictEvent.kwarg_key not implemented")

    @abstractmethod
    def get_kwarg_enable(self) -> Dict[str, Any]:
        """ Enable restriction and return required enable kwarg. """
        raise NotImplementedError("RestrictEvent.enable not implemented")

    @abstractmethod
    def get_kwarg_disable(self) -> Dict[str, Any]:
        """ Disable restriction and return required disable kwarg. """
        raise NotImplementedError("RestrictEvent.disable not implemented")

    @classmethod
    def add_guild(cls, guild_id: int) -> None:
        """ Add guild to event dict. """
        cls.events[guild_id] = {}

    @classmethod
    def remove_guild(cls, guild_id: int) -> None:
        """ Remove guild from event dict. """
        del cls.events[guild_id]

    @classmethod
    def check_guild(cls, guild_id: int) -> bool:
        """ Verify if events dict has guild by id. """
        return guild_id in cls.events

    @classmethod
    def get_guild(cls, guild_id: int) -> Dict[int, RestrictEvent]:
        """ Get guild from event dict. """
        return cls.events[guild_id]

    @classmethod
    def add_event(cls, guild_id: int, event: RestrictEvent) -> None:
        """ Add event to event guild dict. """
        cls.events[guild_id][event.member.id] = event

    @classmethod
    def remove_event(cls, guild_id: int, event_id: int) -> None:
        """ Remove event from event guild dict. """
        del cls.events[guild_id][event_id]

    @classmethod
    def check_event(cls, guild_id: int, event_id: int) -> bool:
        """ Verify if event guild dict has member by id. """
        return event_id in cls.events[guild_id]

    @classmethod
    def get_event(cls, guild_id: int, event_id: int) -> RestrictEvent:
        """ Get event from event guild dict. """
        return cls.events[guild_id][event_id]

    @classmethod
    def get_formatted_list(cls, guild_id: int) -> str:
        """ Get formatted list of running mute events in a guild. """
        events: List[RestrictEvent] = sorted(cls.get_events(guild_id),
                                             key=lambda event: event.member.name.lower())
        if len(events) == 0:
            return f"There are no {cls.name_past} users"

        member_pluralized: str = pluralize(len(events), "member", "members")

        header: str = f"{len(events)} {cls.name_past} {member_pluralized}\n\n"
        name_lengths: Set[int] = {len(event.member.name) for event in events}
        highest_name_length: int = max(name_lengths)

        events_data: List[str] = [(f"{event.member.name.ljust(highest_name_length)}: " +
                                   f"{event.get_remaining_seconds()} seconds")
                                  for event in events]

        formatted_events: str = "\n".join(events_data)

        return format_code_block(header + formatted_events)

    @classmethod
    def get_events(cls, guild_id: int) -> ValuesView[RestrictEvent]:
        """ Get list of events in a given guild. """
        return cls.events[guild_id].values()


class AmputateEvent(RestrictEvent):
    """ Amputate event on progress. """
    events: Dict[int, Dict[int, RestrictEvent]] = None

    name_present: str = "amputate"
    name_past: str = "amputated"
    function_role: Callable = lambda role: role.permissions.send_messages
    kwarg_key: str = "roles"

    def __init__(self, guild: discord.Guild, member: discord.Member, seconds: int):
        super().__init__(member, seconds)
        AmputateEvent.add_event(guild.id, self)

        self.previous_roles: List[discord.Role] = self.member.roles

    def get_kwarg_enable(self) -> Dict[str, List[discord.Role]]:
        """ Enable restriction and return required enable kwarg. """
        return {self.kwarg_key: []}

    def get_kwarg_disable(self) -> Dict[str, List[discord.Role]]:
        """ Disable restriction and return required disable kwarg. """
        return {self.kwarg_key: self.previous_roles}


class DeafEvent(RestrictEvent):
    """ Deaf event on progress. """
    events: Dict[int, Dict[int, RestrictEvent]] = None

    name_present: str = "deaf"
    name_past: str = "deafened"
    function_role: Callable = lambda role: role.permissions.deafen_members
    kwarg_key: str = "deafen"

    def __init__(self, guild: discord.Guild, member: discord.Member, seconds: int):
        super().__init__(member, seconds)
        DeafEvent.add_event(guild.id, self)

    def get_kwarg_enable(self) -> Dict[str, bool]:
        """ Enable restriction and return required enable kwarg. """
        return {self.kwarg_key: True}

    def get_kwarg_disable(self) -> Dict[str, bool]:
        """ Disable restriction and return required disable kwarg. """
        return {self.kwarg_key: False}


class MuteEvent(RestrictEvent):
    """ Mute event on progress. """
    events: Dict[int, Dict[int, RestrictEvent]] = None

    name_present: str = "mute"
    name_past: str = "muted"
    function_role: Callable = lambda role: role.permissions.mute_members
    kwarg_key: str = "mute"

    def __init__(self, guild: discord.Guild, member: discord.Member, seconds: int):
        super().__init__(member, seconds)
        MuteEvent.add_event(guild.id, self)

    def get_kwarg_enable(self) -> Dict[str, bool]:
        """ Enable restriction and return required enable kwarg. """
        return {self.kwarg_key: True}

    def get_kwarg_disable(self) -> Dict[str, bool]:
        """ Disable restriction and return required disable kwarg. """
        return {self.kwarg_key: False}


class Video:
    """ Represent a video to be played. """
    ids: Dict[int, int] = {}
    queues: Dict[int, Dict[int, Video]] = {}

    # Last play/unpause timestamp in each guild
    last_video_plays: Dict[int, float] = {}
    # Last pause timestamp in each guild
    last_video_pauses: Dict[int, float] = {}
    # Last play/unpause remaining duration in each guild
    last_video_remaining_durations: Dict[int, int] = {}

    def __init__(self, title: str, duration: int, guild: discord.Guild, audio_source: discord.FFmpegPCMAudio):

        for char_code in big_char_codes:
            char: str = chr(char_code)
            title = title.replace(char, " ")
        partial_title: str = title if len(title) <= 50 else f"{title[: 47]}..."

        self.id: int = Video.ids[guild.id]
        self.partial_title: str = partial_title
        self.title: str = title
        self.duration: int = duration
        self.guild: discord.Guild = guild
        self.audio_source: discord.FFmpegPCMAudio = audio_source

        Video.ids[guild.id] += 1
        Video.add_video(guild.id, self)

    def get_formatted_duration(self) -> str:
        """ Get formatted representation of duration. """
        return get_formatted_duration(self.duration, justify=True)

    @classmethod
    def get_remaining_duration(cls, guild: discord.Guild) -> int:
        """ Get remaining duration for current video. """
        last_video_play: float = Video.last_video_plays[guild.id]
        last_video_pause: float = Video.last_video_pauses[guild.id]
        last_video_remaining_duration: int = Video.last_video_remaining_durations[guild.id]

        current_time: float = time.time()
        last_pause_or_play: float = max(last_video_play, last_video_pause)
        time_passed: float = current_time - last_pause_or_play

        if guild.voice_client.is_playing():
            remaining_duration: int = int(last_video_remaining_duration - time_passed)
        else:
            remaining_duration: int = int(last_video_remaining_duration)

        return remaining_duration

    @classmethod
    def get_formatted_remaining_duration(cls, guild: discord.Guild) -> str:
        """ Get formatted representation of remaining duration for current video. """
        remaining_duration: int = Video.get_remaining_duration(guild)
        return get_formatted_duration(remaining_duration, justify=False)

    @classmethod
    def get_total_formatted_remaining_duration(cls, guild: discord.Guild) -> str:
        """ Get formatted representation of remaining duration for videos in queue. """
        current_remaining_duration: int = Video.get_remaining_duration(guild)

        videos: List[Video] = Video.get_videos(guild.id)
        next_videos: List[Video] = videos[1:]
        next_videos_duration: Generator[int] = (video.duration for video in next_videos)
        total_next_videos_duration: int = sum(next_videos_duration)

        total_remaining_duration: int = current_remaining_duration + total_next_videos_duration

        return get_formatted_duration(total_remaining_duration, justify=False)

    @classmethod
    def add_queue(cls, guild_id: int) -> None:
        """ Add video dict to guild dict. """
        cls.queues[guild_id] = {}

    @classmethod
    def remove_queue(cls, guild_id: int) -> None:
        """ Remove video dict from guild dict. """
        del cls.queues[guild_id]

    @classmethod
    def get_queue(cls, guild_id: int) -> Dict[int, Video]:
        """ Get video dict from guild dict. """
        return cls.queues[guild_id]

    @classmethod
    def set_queue(cls, guild_id: int, queue: Dict[int, Video]) -> None:
        """ Get video dict from guild dict. """
        cls.queues[guild_id] = queue

    @classmethod
    def clear_queue(cls, guild_id: int) -> None:
        """ Clear video dict for a guild. """
        cls.queues[guild_id] = {}

    @classmethod
    def add_video(cls, guild_id: int, video: Video) -> None:
        """ Add video to video dict. """
        cls.queues[guild_id][video.id] = video

    @classmethod
    def check_video(cls, guild_id: int, video_id: int) -> bool:
        """ Verify if video exists in video dict. """
        return video_id in cls.queues[guild_id]

    @classmethod
    def remove_video(cls, guild_id: int, video_id: int) -> None:
        """ Remove video from video dict. """
        del cls.queues[guild_id][video_id]

    @classmethod
    def remove_next_video(cls, guild_id: int) -> None:
        """ Remove first video from video dict. """
        queue: Dict[int, Video] = cls.queues[guild_id]
        next_video_id: int = next(iter(queue))
        cls.remove_video(guild_id, next_video_id)

    @classmethod
    def get_next_video(cls, guild_id: int) -> Video:
        """ Get first video from video dict. """
        queue: Dict[int, Video] = cls.queues[guild_id]
        next_key: int = next(iter(queue))
        return cls.queues[guild_id][next_key]

    @classmethod
    def get_videos(cls, guild_id: int) -> List[Video]:
        """ Get videos from guild dict. """
        queue: Dict[int, Video] = cls.queues[guild_id]
        return list(queue.values())

    @classmethod
    def get_video(cls, guild_id: int, video_id: int) -> Video:
        """ Get video from videos dict. """
        return cls.queues[guild_id][video_id]

    @classmethod
    def get_formatted_queue(cls, guild: discord.Guild) -> str:
        """ Get formatted queue of videos in guild queue. """
        videos: List[Video] = cls.get_videos(guild.id)

        if len(videos) == 0:
            return "Queue is empty"

        voice_client: discord.VoiceClient = guild.voice_client

        video_pluralized: str = pluralize(len(videos), "video", "videos")

        is_playing: bool = voice_client.is_playing()
        is_paused: bool = voice_client.is_paused()
        video_state: str = "Playing" if is_playing else "Paused" if is_paused else "Downloading"

        id_header: str = "ID"
        title_header: str = "Title"
        duration_header: str = "Duration"

        next_id: int = Video.ids[guild.id]
        highest_id: int = next_id - 1
        id_field_length: int = max(len(str(highest_id)), len(id_header))

        title_lengths: Set[int] = {len(video.partial_title) for video in videos}
        title_field_length: int = max(*title_lengths, len(title_header))

        duration_lengths: Set[int] = {len(video.get_formatted_duration()) for video in videos}
        duration_field_length: int = max(*duration_lengths, len(duration_header))

        formatted_remaining_duration: str = Video.get_formatted_remaining_duration(guild)

        formatted_total_remaining_duration: str = Video.get_total_formatted_remaining_duration(guild)

        header: str = (f"{len(videos)} {video_pluralized} found\n" +
                       f"Current state: {video_state}\n" +
                       f"Current remaining: {formatted_remaining_duration}\n" +
                       f"Total remaining: {formatted_total_remaining_duration}\n\n")

        table_header: str = (f"{id_header.ljust(id_field_length)} | " +
                             f"{title_header.ljust(title_field_length)} | " +
                             f"{duration_header.ljust(duration_field_length)}\n")

        table_separator: str = f"{'-' * len(table_header)}\n"

        videos_data: List[str] = [(f"{str(video.id).ljust(id_field_length)} | " +
                                   f"{video.partial_title.ljust(title_field_length)} | " +
                                   f"{video.get_formatted_duration().rjust(duration_field_length)}")
                                  for video in videos]

        # Current video pointer arrow
        videos_data[0] += " <-"

        formatted_videos: str = "\n".join(videos_data)

        return format_code_block(header + table_header + table_separator + formatted_videos)


# ---------- Exceptions ---------- #

class InvalidIntException(Exception):
    """ Exception triggered when a expected integer is invalid. """
    def __init__(self, name: str, min_value: int, max_value: int):
        self.name: str = name
        self.min_value: int = min_value
        self.max_value: int = max_value


class MissingParameterException(Exception):
    """ Exception triggered when a command doesn't receive a required parameter. """
    def __init__(self, parameter_name: str):
        self.parameter_name: str = parameter_name


class TooManyParametersException(Exception):
    """ Exception triggered when a command receives more parameters than expected. """
    pass


class NoPermissionException(Exception):
    """ Exception triggered when a user requests an action which he doesn't have permission to execute. """
    def __init__(self, user_mention: str, action_name: str):
        self.user_mention: str = user_mention
        self.action_name: str = action_name


class UserNotInChannelException(Exception):
    """ Exception triggered when a user requests an action which requires him to be in a voice channel. """
    def __init__(self):
        pass


# ---------- Internal functions ---------- #

def pluralize(length: int, singular: str, plural: str) -> str:
    """ Decide between singular or plural version of a message. """
    return singular if length == 1 else plural


def extract_id(s: str) -> int:
    """ Extract id from given string, return -1 if pattern is invalid. """
    match: re.Match = re.search("\d+", s)
    return int(match[0]) if match is not None else -1


def validate_int(num_str: str, min_value: int, max_value: int) -> bool:
    """ Inform if given number is in range of given boundaries. """
    try:
        is_num_number: bool = re.search("\D", num_str) is None
        num: int = int(num_str)
        is_num_on_bounds: bool = (min_value <= num <= max_value)
        is_num_valid: bool = is_num_number and is_num_on_bounds
    except ValueError:
        is_num_valid: bool = False

    return is_num_valid


def get_pediu() -> str:
    """ Get string "pediu" with randomized changes. """
    s: str = "pediu"

    text: str = "".join([random.choice([char.lower(), char.upper()]) for char in s])
    spaces: str = " " * random.randint(1, 3)
    laughs: str = "".join("k" if random.random() < 0.7 else "j" for _ in range(random.randint(0, 4)))
    exclamations: str = "?" * random.randint(0, 3)
    result: str = text + spaces + laughs + exclamations

    return result


def request_image_srcs(query_param: str, max_page: int = 1) -> List[str]:
    """ Return image srcs found until a given page from query service. """
    base_url: str = "http://results.dogpile.com"

    request_headers: Dict[str, str] = {
        "Accept": "text/html",
        "User-Agent": "Chrome"
    }

    urls: List[str] = [f"{base_url}/serp?qc=images&q={query_param}&page={page}&sc={sc_param}"
                       for page in range(1, max_page + 1)]

    srcs: List[str] = []
    for url in urls:
        response: requests.Response = requests.get(url, headers=request_headers)
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
    # Prevent letter substitution through letters
    s = s.replace("0", "o").replace("1", "i").replace("3", "e").replace("4", "a")
    words: List[str] = re.findall("\S+", s) or []

    bad_words: Set[str] = {word for word in words
                           if any(re.search(f"\\b{bad_word_regex}\\b", word.lower()) is not None
                                  for bad_word_regex in bad_word_regexes)}

    for bad_word in bad_words:
        s = s.replace(bad_word, "#" * len(bad_word))

    bad_word_amount: int = sum([word == bad_word for word in words for bad_word in bad_words])

    return s, bad_word_amount


def check_permissions(author: discord.Member, function_role: Callable) -> bool:
    """ Inform if author has permission to perform a given action. """
    roles: List[discord.Role] = author.roles
    role_has_permission: bool = any([function_role(role) for role in roles])
    is_admin: bool = any([role.permissions.administrator for role in roles])
    is_owner: bool = author.id == author.guild.owner_id
    has_permission: bool = role_has_permission or is_admin or is_owner

    return has_permission


async def restrict(message: discord.Message,
                   parameters: List[str],
                   restrict_event_class: ClassVar[RestrictEvent]) -> str:
    """ Restrict someone for given number of seconds. """
    length: int = len(parameters)
    required_length: int = 3

    # Validate length
    if length == 1:
        raise MissingParameterException("member")
    if length == 2:
        raise MissingParameterException("seconds")
    if length > required_length:
        raise TooManyParametersException()

    member_name: str = parameters[1]
    member_id: int = extract_id(member_name)
    seconds_str: str = parameters[2]

    # Validate member highlight
    is_highlight: bool = member_id != -1
    if not is_highlight:
        return "Target user must be highlighted"

    # Validate author permissions
    author: discord.Member = message.author
    function_role: Callable = restrict_event_class.function_role
    has_permission: bool = check_permissions(author, function_role)

    if not has_permission:
        raise NoPermissionException(author.mention, restrict_event_class.name_present)

    # Get member
    members: List[discord.Member] = [member for member in message.guild.members]
    member: discord.Member = next((member for member in members if member.id == member_id), None)

    # Validate seconds
    min_seconds_amount: int = 1
    max_seconds_amount: int = 7200
    is_seconds_valid: bool = validate_int(seconds_str, min_seconds_amount, max_seconds_amount)

    if not is_seconds_valid:
        raise InvalidIntException("seconds", min_seconds_amount, max_seconds_amount)
    seconds: int = int(seconds_str)

    # Restrict and sleep
    restrict_event: RestrictEvent = restrict_event_class(message.guild, member, seconds)
    reply: str = f"User {member.mention} is {restrict_event_class.name_past} for {seconds} seconds"

    try:
        kwarg_enable: Dict[str, Any] = restrict_event.get_kwarg_enable()
        await member.edit(**kwarg_enable)
        await message.channel.send(reply)
        await asyncio.sleep(seconds)
    except discord.errors.HTTPException as e:
        if restrict_event_class.name_present in voice_restrictions:
            return "Target user is not in voice chat"
        else:
            print(e)
            return (f"User {message.author.mention} doesn't have permission to " +
                    f"{restrict_event_class.name_present} user {member.mention}")

    # Unrestrict if still restricted
    guild_id: int = message.guild.id
    is_restricted: bool = restrict_event_class.check_event(guild_id, member.id)

    if is_restricted:
        current_restrict_event: RestrictEvent = restrict_event_class.get_event(guild_id, member_id)
        is_same_restriction: bool = restrict_event.id == current_restrict_event.id

        if is_same_restriction:
            kwarg_disable: Dict[str, Any] = restrict_event.get_kwarg_disable()
            await member.edit(**kwarg_disable)
            restrict_event_class.remove_event(guild_id, member.id)


async def unrestrict(message: discord.Message,
                     parameters: List[str],
                     restrict_event_class: ClassVar[RestrictEvent]) -> str:
    """ Unrestrict a restricted member. """
    length: int = len(parameters)
    required_length: int = 2

    # Validate length
    if length == 1:
        raise MissingParameterException("member")
    if length > required_length:
        raise TooManyParametersException()

    member_name: str = parameters[1]
    member_id: int = extract_id(member_name)

    # Validate member highlight
    is_highlight: bool = member_id != -1
    if not is_highlight:
        return "Target user must be highlighted"

    # Validate author permissions
    author: discord.Member = message.author
    function_role: Callable = restrict_event_class.function_role
    has_permission: bool = check_permissions(author, function_role)

    if not has_permission:
        raise NoPermissionException(author.mention, restrict_event_class.name_present)

    # Get member
    members: List[discord.Member] = [member for member in message.guild.members]
    member: discord.Member = next((member for member in members if member.id == member_id), None)

    # Unrestrict if still restricted
    guild_id: int = message.guild.id
    is_restricted: bool = restrict_event_class.check_event(guild_id, member.id)
    if is_restricted:
        restrict_event: RestrictEvent = restrict_event_class.get_event(guild_id, member.id)
        kwarg_disable: Dict[str, Any] = restrict_event.get_kwarg_disable()
        await member.edit(**kwarg_disable)

        event_exists: bool = restrict_event_class.check_event(guild_id, member.id)
        if event_exists:
            restrict_event_class.remove_event(guild_id, member.id)
    else:
        return f"User {member.mention} is not time-{restrict_event_class.name_past}"


async def restrictionlist(message: discord.Message,
                          parameters: List[str],
                          restrict_event_class: ClassVar[RestrictEvent]) -> str:
    """ Get list of currently restricted members on requested guild. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    return restrict_event_class.get_formatted_list(message.guild.id)


async def process_message(message: discord.Message) -> str:
    """ Handle message reply. """
    reply: str = None
    sent_by_bot: bool = message.author.bot
    is_empty: bool = message.content == ""

    if not sent_by_bot and not is_empty:

        print(f"{message.guild}: {message.content}")

        # Handle special messages
        content_lower: str = message.content.lower()
        key: str = next((key for key in special_messages if content_lower.startswith(key)), None)
        is_special: bool = key is not None
        if is_special:
            special_function: Callable = special_messages[key]
            special_message: str = special_function()
            return special_message

        # Verify if message is just bot highlight
        bot_was_highlighted: bool = re.search(f"^<@!?{bot_id}>$", message.content) is not None

        # Handle input
        parameters: List[str] = re.findall("\S+", message.content)
        command: str = parameters[0]
        is_help: bool = re.search(f"^{prefix}help", message.content) is not None
        command_exists: bool = command in commands_map

        # Reply highlight
        if bot_was_highlighted:
            reply = suggest_help()
        # Reply to help
        elif is_help:
            reply = get_help(parameters)
        # Run command
        elif command_exists:
            command_function: Callable = commands_map[command]
            reply = await command_function(message, parameters)

    return reply


def check_spam(text: str) -> bool:
    """ Inform if text received is spam. """
    # Require minimum amount of distinct chars
    min_distinct_chars_length: int = 5

    distinct_chars: Set[str] = set(text)
    distinct_chars_length: int = len(distinct_chars)
    if distinct_chars_length < min_distinct_chars_length:
        return True

    # Require lowercase letters
    has_lower: bool = any(char.islower() for char in distinct_chars)
    if not has_lower:
        return True

    # Attempt to find recurring pattern
    has_recurring_pattern: bool = any(text[: length] == text[length: length * 2]
                                      for length in range(4, len(text) // 2 + 1))
    if has_recurring_pattern:
        return True

    # Require some chars to be letters
    min_letter_percentage: int = 40

    letters: List[str] = [char for char in text if char.isalpha()]
    letter_percentage: float = (len(letters) / len(text)) * 100
    if letter_percentage < min_letter_percentage:
        return True

    # Require some chars to be spaces
    min_space_percentage: int = 5

    spaces: List[str] = [char for char in text if char == " "]
    space_percentage: float = (len(spaces) / len(text)) * 100
    if space_percentage < min_space_percentage:
        return True

    return False


async def request_video_url(query: str) -> str:
    """ Find the most relevant youtube video url for a given query. """
    query_url: str = ("https://content.googleapis.com/youtube/v3/search"
                      f"?maxResults=1&part=snippet&q={query}&key={api_key}&type=video")

    async with aiohttp.ClientSession() as session:
        async with session.get(query_url) as response:
            response_obj: Dict = await response.json()
            video_id: str = response_obj["items"][0]["id"]["videoId"]

    video_url: str = f"https://www.youtube.com/watch?v={video_id}"

    return video_url


async def request_voice_client(author: discord.Member, guild: discord.Guild) -> discord.VoiceClient:
    """ Request voice client for given guild and connect bot if it's not connected. """
    is_bot_in_channel: bool = guild.voice_client is not None
    if is_bot_in_channel:
        voice_client: discord.VoiceClient = guild.voice_client
    else:
        voice_client: discord.VoiceClient = await author.voice.channel.connect()

    return voice_client


async def disconnect(guild: discord.Guild) -> None:
    """ Disconnect voice client from guild and clear queue. """
    await guild.voice_client.disconnect()
    Video.clear_queue(guild.id)


async def update_queue(guild: discord.Guild, is_skip=False) -> None:
    """ Handle video add/play. """
    queue: Dict[int, Video] = Video.get_queue(guild.id)
    is_play: bool = len(queue) == 1

    try:
        if is_skip:
            Video.remove_next_video(guild.id)
            guild.voice_client.stop()
        next_video: Video = Video.get_next_video(guild.id)
        if not guild.voice_client.is_playing():
            guild.voice_client.play(next_video.audio_source,
                                    after=lambda e: loop.create_task(update_queue(guild, is_skip=True)))

        # Update remaining duration on video play / skip and last play/pause timestamp
        if is_play or is_skip:
            Video.last_video_plays[guild.id] = time.time()
            Video.last_video_remaining_durations[guild.id] = next_video.duration

    except StopIteration as e:
        await disconnect(guild)


def shuffle_dict(d: Dict[int, Any]) -> Dict[int, Any]:
    """ Shuffle a given dict. """
    keys: List[int] = list(d.keys())
    random.shuffle(keys)
    shuffled_d: Dict[int, Any] = {key: d[key] for key in keys}

    return shuffled_d


def count_big_chars(s: str) -> int:
    """ Count big chars in a given string. """
    return sum(ord(char) in big_char_codes for char in s)


async def get_current_datetime() -> datetime.datetime:
    """ Request current UTC time and get datetime object from it. """
    url: str = "https://www.epochconverter.com"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content: str = await response.text()
            soup: bs4.BeautifulSoup = bs4.BeautifulSoup(content, features="lxml")
            element: bs4.element.Tag = soup.select_one("#ecclock")
            timestamp: int = int(element.text)

    return datetime.datetime.fromtimestamp(timestamp).now() + datetime.timedelta(hours=0)


def get_formatted_duration(seconds: int, justify=False) -> str:
    """ Get formatted representation of duration. """
    formatted_hours: str = str(seconds // 3600)
    formatted_minutes: str = str((seconds // 60) % 60)
    formatted_seconds: str = str(seconds % 60)

    if justify:
        formatted_minutes = formatted_minutes.rjust(2)
        formatted_seconds = formatted_seconds.rjust(2)

    if seconds < 60:
        return f"{formatted_seconds}s"
    elif seconds < 3600:
        return f"{formatted_minutes}m {formatted_seconds}s"
    else:
        return f"{formatted_hours}h {formatted_minutes}m {formatted_seconds}s"


async def update_queue_and_feedback(guild: discord.guild, video: Video) -> str:
    """ Update queue and provide feedback about last video appended. """
    queue: Dict[int, Video] = Video.get_queue(guild.id)
    state: str = "Playing" if len(queue) == 1 else "Queued"
    await update_queue(guild)

    return f"{state} {video.title}"


# ---------- Interface ---------- #


def get_help(parameters: List[str]) -> str:
    """ Describe all commands briefly or a given command extensively. """
    length: int = len(parameters)
    required_lengths: Set[int] = {1, 2}
    max_required_length: int = max(required_lengths)
    is_general_help: bool = length == 1
    is_command_help: bool = length == 2

    if length > max_required_length:
        raise TooManyParametersException()

    # General help
    if is_general_help:
        header: str = "List of available commands:\n\n"
        commands_data: List[str] = [command.get_brief_data() for command in commands.values()]
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


async def code(message: discord.Message, parameters: List[str]) -> None:
    """ Provide file with current source code. """
    length: int = len(parameters)
    required_length: int = 1

    if length > required_length:
        raise TooManyParametersException()

    code_file: discord.File = discord.File("./bot.py")
    await message.channel.send(file=code_file)


async def report(message: discord.Message, parameters: List[str]) -> str:
    """ Send a message to be logged. """
    length: int = len(parameters)
    min_required_length: int = 2

    if length < min_required_length:
        raise MissingParameterException("message")

    content: str = message.content
    report_message_match: re.Match = re.search(f"{prefix}report\s+", content)
    report_message_index: int = report_message_match.end()
    report_message: str = content[report_message_index:]

    is_duplicate: bool = report_message in reports
    is_spam: bool = check_spam(report_message)

    # Check bad words
    replaced_content: str
    bad_word_amount: int
    replaced_content, bad_word_amount = replace_bad_words(message.content)
    has_bad_word: bool = bad_word_amount > 0

    if is_duplicate:
        return "This message was already sent"
    if is_spam:
        return "Your message was detected as spam and got filtered"
    if has_bad_word:
        return "Your message has bad words and got filtered"

    reports.add(report_message)
    with open(report_path, "w") as file:
        reports_json: str = json.dumps(list(reports), indent=4)
        file.write(reports_json)
    return "Thank you for helping!"


async def amputate(message: discord.Message, parameters: List[str]) -> str:
    """ Amputate someone for given number of seconds. """
    return await restrict(message, parameters, AmputateEvent)


async def unamputate(message: discord.Message, parameters: List[str]) -> str:
    """ Unamputate a amputated member. """
    return await unrestrict(message, parameters, AmputateEvent)


async def amputatelist(message: discord.Message, parameters: List[str]) -> str:
    """ Get list of currently amputated members on requested guild. """
    return await restrictionlist(message, parameters, AmputateEvent)


async def deaf(message: discord.Message, parameters: List[str]) -> str:
    """ Deafen someone for given number of seconds. """
    return await restrict(message, parameters, DeafEvent)


async def undeaf(message: discord.Message, parameters: List[str]) -> str:
    """ Undeaf a deafened member. """
    return await unrestrict(message, parameters, DeafEvent)


async def deaflist(message: discord.Message, parameters: List[str]) -> str:
    """ Get list of currently deafened members on requested guild. """
    return await restrictionlist(message, parameters, DeafEvent)


async def mute(message: discord.Message, parameters: List[str]) -> str:
    """ Mute someone for given number of seconds. """
    return await restrict(message, parameters, MuteEvent)


async def unmute(message: discord.Message, parameters: List[str]) -> str:
    """ Unmute a muted member. """
    return await unrestrict(message, parameters, MuteEvent)


async def mutelist(message: discord.Message, parameters: List[str]) -> str:
    """ Get list of currently muted members on requested guild. """
    return await restrictionlist(message, parameters, MuteEvent)


async def serverlist(message: discord.Message, parameters: List[str]) -> str:
    """ Request list of servers in which TPoseBot is present. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    guilds: List[discord.Guild] = sorted([guild for guild in client.guilds if guild is not None],
                                         key=lambda guild: guild.name.lower())
    header: str = f"{len(guilds)} servers found\n\n"

    name_lengths: Set[int] = {len(guild.name) for guild in guilds}
    highest_name_length: int = max(name_lengths)

    guilds_data: List[str] = [f"{guild.name.ljust(highest_name_length)} | " +
                              f"Member count: {len(guild.members)}"
                              for guild in guilds]
    formatted_guilds: str = "\n".join(guilds_data)

    return format_code_block(header + formatted_guilds)


async def tpose(message: discord.Message, parameters: List[str]) -> str:
    """ Request random tpose image. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    src: str = random.choice(srcs)

    return src


async def cursed(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to join voice channel and play a random cursed audio. """
    length: int = len(parameters)
    required_length: int = 1

    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    filenames: List[str] = os.listdir(f"./{cursed_audios_dir}")
    filename: str = random.choice(filenames)
    file_path = f"./{cursed_audios_dir}/{filename}"

    duration: int = int(librosa.get_duration(filename=file_path))
    audio_source: discord.FFmpegPCMAudio = discord.FFmpegPCMAudio(file_path)

    video: Video = Video(filename, duration, message.guild, audio_source)
    result: str = await update_queue_and_feedback(message.guild, video)

    return result


async def play(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to join voice channel and optionally play a song. """
    length: int = len(parameters)

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    has_query: bool = length > 1
    query: str = " ".join(parameters[1:]) if has_query else None

    # Search for a video if query was given
    if query is not None:

        youtube_video_regex: str = "youtube.com/watch\?.*v=[\w\-]{11}"
        is_query_url: bool = re.search(youtube_video_regex, query) is not None
        url: str = query if is_query_url else await request_video_url(query)

        video_id: str = re.search("(?<=v=)[\w\-]{11}", url)[0]

        with youtube_dl.YoutubeDL() as ydl:
            video_info: Dict = await loop.run_in_executor(None,
                                                          lambda: ydl.extract_info(url,
                                                                                   download=False))

        video_title: str = video_info["title"]
        video_duration: int = int(video_info["duration"])

        if video_duration > 3600:
            return "This video is too large"

        source_file_path: str = f"./{youtube_videos_source_dir}/{video_id}"
        source_file_exists: bool = os.path.exists(source_file_path)

        if not source_file_exists:
            await message.channel.send(f"Downloading {video_title}...")

            ydl_opts = {
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredquality": "192"
                    }
                ],
                "outtmpl": f"{youtube_videos_download_dir}/{video_id}"
            }

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                await loop.run_in_executor(None,
                                           lambda: ydl.download([url]))

            filenames: Generator[str] = (filename for filename in os.listdir(youtube_videos_download_dir))
            filename: str = next(filename for filename in filenames
                                 if filename.find(video_id) > -1)
            os.rename(f"./{youtube_videos_download_dir}/{filename}", source_file_path)

        audio_source: discord.FFmpegPCMAudio = discord.FFmpegPCMAudio(source_file_path)

        # Ensure voice client is still connected
        voice_client = await request_voice_client(message.author, message.guild)

        video: Video = Video(video_title, video_duration, message.guild, audio_source)
        result: str = await update_queue_and_feedback(message.guild, video)

        return result


async def leave(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to leave voice channel. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    guild: discord.Guild = message.guild
    is_in_voice_channel: bool = guild.voice_client is not None

    if is_in_voice_channel:
        await disconnect(guild)
    else:
        return "I am not connected to a voice channel"


async def pause(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to pause current video. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    if voice_client.is_playing():
        voice_client.pause()

        # Update last video pause
        Video.last_video_pauses[message.guild.id] = time.time()

        # Update remaining duration
        last_video_play: float = Video.last_video_plays[message.guild.id]
        last_video_pause: float = Video.last_video_pauses[message.guild.id]
        last_video_remaining_duration: int = Video.last_video_remaining_durations[message.guild.id]

        time_passed: int = last_video_pause - last_video_play
        Video.last_video_remaining_durations[message.guild.id] = last_video_remaining_duration - time_passed
    else:
        return "I am not playing already"


async def unpause(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to unpause current video. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    if voice_client.is_playing():
        return "I am already unpaused"
    else:
        voice_client.resume()

        # Update last video play
        Video.last_video_plays[message.guild.id] = time.time()


async def stop(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to stop playing and reset queue. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    voice_client.stop()
    Video.clear_queue(message.guild.id)


async def skip(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to skip current video. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    if voice_client.is_playing():
        message.guild.voice_client.stop()
    else:
        return "There are no videos to skip"


async def remove(message: discord.Message, parameters: List[str]) -> str:
    """ Request bot to remove video with given id. """
    length: int = len(parameters)
    required_length: int = 2

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    is_user_in_channel: bool = message.author.voice is not None
    if not is_user_in_channel:
        raise UserNotInChannelException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    # Check if id is number
    video_id_str = parameters[1]
    try:
        video_id: int = int(video_id_str)
    except ValueError:
        return "Given video id is not a number"

    # Attempt to remove video by id
    try:
        video: Video = Video.get_video(message.guild.id)
        is_current_video: bool = video.id == video_id

        if is_current_video:
            await update_queue(message.guild, skip=True)
        else:
            Video.remove_video(message.guild.id, video_id)
    except KeyError:
        return "There is no video with given id"

    return f"{video.title} was removed from queue"


async def shuffle(message: discord.Message, parameters: List[str]) -> str:
    """ Shuffle videos in queue. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)

    queue: Dict[int, Video] = Video.get_queue(message.guild.id)

    if len(queue) < 3:
        return "The queue must have 3 or more videos to be shuffable"

    current_key: int = next(iter(queue))
    current_video: Video = queue[current_key]
    next_videos: Dict[int, Video] = {key: value for key, value in queue.items() if key != current_key}

    shuffled_next_videos: Dict[int, Video] = shuffle_dict(next_videos)
    shuffled_queue: Dict[int, Video] = {current_key: current_video, **shuffled_next_videos}

    Video.set_queue(message.guild.id, shuffled_queue)
    return "Queue just got shuffled"


async def queue(message: discord.Message, parameters: List[str]) -> str:
    """ Get current video queue. """
    length: int = len(parameters)
    required_length: int = 1

    # Validate length
    if length > required_length:
        raise TooManyParametersException()

    formatted_queue: str = Video.get_formatted_queue(message.guild)

    return formatted_queue


async def dice(message: discord.Message, parameters: List[str]) -> str:
    """ Select a random number between 1 and given max number. """
    length: int = len(parameters)
    required_length: int = 2

    # Validate length
    if length == 1:
        raise MissingParameterException("max number")

    if length > required_length:
        raise TooManyParametersException()

    # Validate number
    max_num_str: str = parameters[1]
    min_value: int = 2
    max_value: int = 1000000
    is_max_num_valid: int = validate_int(max_num_str, min_value, max_value)

    if not is_max_num_valid:
        raise InvalidIntException("max number", min_value, max_value)
        return ("Invalid max number, it has to be an integer " +
                f"between {min_value} and {max_value}")

    min_num: int = 1
    max_num: int = int(max_num_str)
    random_num: int = random.randint(1, max_num)

    # Highlight if max num was achieved
    result: str = str(random_num)
    if random_num in {min_num, max_num}:
        return format_bold_text(result)

    return result


async def wipe(message: discord.Message, parameters: List[str]) -> str:
    """ Remove all messages sent within last given number of seconds. """
    length: int = len(parameters)
    required_length: int = 2

    # Validate length
    if length == 1:
        raise MissingParameterException("seconds")

    if length > required_length:
        raise TooManyParametersException()

    # Validate author permissions
    author: discord.Member = message.author
    function_role: Callable = lambda role: role.permissions.manage_messages
    has_permission: bool = check_permissions(author, function_role)

    if not has_permission:
        action_name: str = "wipe"
        raise NoPermissionException(author.mention, action_name)

    # Validate number
    total_seconds_str: str = parameters[1]
    min_value: int = 1
    max_value: int = 7200
    is_max_num_valid: int = validate_int(total_seconds_str, min_value, max_value)

    if not is_max_num_valid:
        raise InvalidIntException("seconds", min_value, max_value)

    total_seconds: int = int(total_seconds_str)

    now: datetime.datetime = await get_current_datetime()
    date: datetime.datetime = now - datetime.timedelta(seconds=total_seconds)
    message_amount: int = 0
    limit: int = 500

    async for recent_message in message.channel.history(limit=limit, after=date, oldest_first=False):
        await recent_message.delete()
        message_amount += 1

    message_pluralized: str = pluralize(message_amount, "message", "messages")
    seconds_pluralized: str = pluralize(total_seconds, "second", "seconds")

    result: str = (f"I just wiped {message_amount} {message_pluralized} " +
                   f"sent within the last {total_seconds} {seconds_pluralized}")

    if message_amount == limit:
        result += "That's the highest amount of messages I can delete for each wipe call"

    return result


async def nword(message: discord.Message, parameters: List[str]) -> str:
    """ Get a random word that starts with n. """
    length: int = len(parameters)
    required_length: int = 1

    if length > required_length:
        raise TooManyParametersException()

    n_word: str = random.choice(n_words)

    return n_word


# ---------- Application variables ---------- #

def TODO():
    """ T. """
    url: str = "https://ytcutter.com/ytcutter.php?a=form1"
    form_body: Dict[str, str] = {
        "videoId": "usXOS6vZOu8",
        "startTime": "62.7",
        "endTime": "68.8",
        "quality": "hd720",
        "format": "audio"
    }


# Async scheduler
loop: asyncio.ProactorEventLoop = asyncio.get_event_loop()

# Tpose image srcs
srcs: List[str] = request_tpose_srcs(max_page=3)

# Char code for chars that become bigger in code blocks
big_char_codes: Set[int] = {12300, 12301, 12302, 12303}

# Map string to command object
commands: Dict[str, Command] = {
    "help": Command(f"{prefix}help",
                    ("Provide brief description for all commands, " +
                     "or extended description for a specific command " +
                     "if any is given"),
                    f"Get help for a command\n{prefix}help mute"),
    "code": Command(f"{prefix}code",
                    "Provide file with my current source code",
                    f"\n{prefix}code"),
    "report": Command(f"{prefix}report",
                      "Send a message to developer (report a bug, request a feature, etc)",
                      (f"Report a permission related bug" +
                       f"\n{prefix}report administrators are not having permission to mute")),
    "amputate": Command(f"{prefix}amputate",
                        ("Amputate user for a given number of seconds, removing all attached roles, " +
                         "event is aborted if target member roles are updated while it runs"),
                        f"Amputate Fred for 30 seconds\n{prefix}amputate @Fred 30"),
    "unamputate": Command(f"{prefix}unamputate",
                          "Unamputate a amputated user",
                          f"Unamputate Fred\n{prefix}unamputate @Fred"),
    "amputatelist": Command(f"{prefix}amputatelist",
                            "Get list of currently amputated users",
                            f"\n{prefix}amputatelist"),
    "mute": Command(f"{prefix}mute",
                    "Mute user for a given number of seconds",
                    f"Mute Fred for 30 seconds\n{prefix}mute @Fred 30"),
    "unmute": Command(f"{prefix}unmute",
                      "Unmute a muted user",
                      f"Unmute Fred\n{prefix}unmute @Fred"),
    "mutelist": Command(f"{prefix}mutelist",
                        "Get list of currently muted users",
                        f"\n{prefix}mutelist"),
    "deaf": Command(f"{prefix}deaf",
                    "Deafen user for a given number of seconds",
                    f"Deafen Fred for 30 seconds\n{prefix}deaf @Fred 30"),
    "undeaf": Command(f"{prefix}undeaf",
                      "Undeaf a deafened user",
                      f"Undeaf Fred\n{prefix}undeaf @Fred"),
    "deaflist": Command(f"{prefix}deaflist",
                        "Get list of currently deafened users",
                        f"\n{prefix}deaflist"),
    "serverlist": Command(f"{prefix}serverlist",
                          "Get list of servers in which I am present",
                          f"\n{prefix}serverlist"),
    "tpose": Command(f"{prefix}tpose",
                     "Get random tpose image",
                     f"\n{prefix}tpose"),
    "cursed": Command(f"{prefix}cursed",
                      "Play a random cursed audio... beware",
                      f"\n{prefix}cursed"),
    "play": Command(f"{prefix}play",
                    "Request me to join voice channel and optionally search for a video to play",
                    f"Search for a tpose related video to play\n{prefix}play tpose"),
    "leave": Command(f"{prefix}leave",
                     "Request me to leave voice channel",
                     f"\n{prefix}leave"),
    "pause": Command(f"{prefix}pause",
                     "Pause current video",
                     f"\n{prefix}pause"),
    "unpause": Command(f"{prefix}unpause",
                       "Unpause current video",
                       f"\n{prefix}unpause"),
    "stop": Command(f"{prefix}stop",
                    "Request me to stop playing",
                    f"\n{prefix}stop"),
    "skip": Command(f"{prefix}skip",
                    "Skip current video",
                    f"\n{prefix}skip"),
    "remove": Command(f"{prefix}remove",
                      "Remove video by id",
                      f"Remove video with id 3\n{prefix}remove 3"),
    "shuffle": Command(f"{prefix}shuffle",
                       "Randomly shuffle videos in queue",
                       f"\n{prefix}shuffle"),
    "queue": Command(f"{prefix}queue",
                     "Get current video queue",
                     f"\n{prefix}queue"),
    "dice": Command(f"{prefix}dice",
                    "Roll a dice that returns a random value from 1 until given number",
                    f"Roll a random number from 1 to 10\n{prefix}roll 10"),
    "wipe": Command(f"{prefix}wipe",
                    "Remove all messages sent within last given number of seconds",
                    f"Remove all messages sent within last 30 seconds\n{prefix}wipe 30"),
    "wipe": Command(f"{prefix}nword",
                    "Get a random word that starts with n",
                    f"\n{prefix}nword")
}

# Specific messages to be replied
special_messages: Dict[str, Callable] = {
    "quem": get_pediu,
    "ninguem": lambda: "pediu",
    "ok": lambda: "boomer",
    "comedores de": lambda: "coc\u00f4",
    "oi": lambda: "oi",
    "que": lambda: "ijo"
}

commands_map: Dict[str, Callable] = {
    f"{prefix}help": get_help,
    f"{prefix}code": code,
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
    f"{prefix}tpose": tpose,
    f"{prefix}cursed": cursed,
    f"{prefix}play": play,
    f"{prefix}leave": leave,
    f"{prefix}pause": pause,
    f"{prefix}unpause": unpause,
    f"{prefix}stop": stop,
    f"{prefix}skip": skip,
    f"{prefix}remove": remove,
    f"{prefix}shuffle": shuffle,
    f"{prefix}queue": queue,
    f"{prefix}dice": dice,
    f"{prefix}wipe": wipe,
    f"{prefix}nword": nword
}

voice_restrictions: Set[str] = {"deaf", "mute"}


# ---------- Event listeners ---------- #

# Bot connect
@client.event
async def on_connect():
    # Initialize data that requires connection
    RestrictEvent.ids = {guild.id: 1 for guild in client.guilds}
    RestrictEvent.events = {guild.id: {} for guild in client.guilds}

    for restrict_event_class in RestrictEvent.__subclasses__():
        restrict_event_class.ids = copy.deepcopy(RestrictEvent.ids)
        restrict_event_class.events = copy.deepcopy(RestrictEvent.events)

    Video.ids = {guild.id: 1 for guild in client.guilds}
    Video.queues = {guild.id: {} for guild in client.guilds}
    Video.last_video_plays = {guild.id: 0.0 for guild in client.guilds}
    Video.last_video_pauses = {guild.id: 0.0 for guild in client.guilds}
    Video.last_video_remaining_durations = {guild.id: 0.0 for guild in client.guilds}


# Bot ready
@client.event
async def on_ready():
    print(f"{client.user} awoke")


# Send message
@client.event
async def on_message(message: discord.Message):
    try:
        channel: discord.TextChannel = message.channel

        reply: str = await process_message(message)
        if reply is not None:
            await channel.send(reply)
    except InvalidIntException as e:
        await channel.send(f"Invalid {e.name}, it has to be an integer between {e.min_value} and {e.max_value}")
    except MissingParameterException as e:
        await channel.send(f"No {e.parameter_name} parameter was given")
    except TooManyParametersException as e:
        await channel.send("Too many parameters")
    except NoPermissionException as e:
        await channel.send(f"User {e.user_mention} doesn't have permission to {e.action_name}")
    except UserNotInChannelException as e:
        await channel.send("You must be connected to a voice channel to use this command")
    except UnicodeEncodeError:
        pass

    # START BURRICE SECTION
    # If guild is decente
    decente_guild_id: int = 289874563230072846

    if message.guild.id == decente_guild_id:

        fred_member_id: int = 239388097714978817
        os_fodas_role_id: int = 549369366584754207

        os_fodas_role: discord.Role = discord.utils.find(lambda role: role.id == os_fodas_role_id, message.guild.roles)

        if message.content == "--goiaba":

            author: discord.Member = message.author
            voice_channel: discord.VoiceChannel = message.author.voice.channel
            if os_fodas_role in message.author.roles or message.author.id == fred_member_id:

                # Explosive goiaba audio
                voice_client: discord.VoiceClient = await request_voice_client(message.author, message.guild)
                file_path: str = f"{base_path}/ruim.mp3"
                audio_source: discord.FFmpegPCMAudio = discord.FFmpegPCMAudio(file_path)
                video = Video(file_path, 99999, message.guild, audio_source)

                # Mute everyone from voice channel except the person who called it
                for member in voice_channel.members:
                    if member.id not in [author.id, bot_id]:
                        await member.edit(mute=True)

                await update_queue(message.guild)
                await message.channel.send("XDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
            else:
                await message.channel.send(f"This command requires role {os_fodas_role.name}")
    # END BURRICE SECTION


# Join, leave, mute, deafen on VC
@client.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState,
                                after: discord.VoiceState):
    was_unmuted: bool = before.mute and not after.mute
    was_undeafen: bool = before.deaf and not after.deaf
    guild_id: int = member.guild.id

    # Remove dict element on unmute
    if was_unmuted:
        if MuteEvent.check_event(guild_id, member.id):
            MuteEvent.remove_event(guild_id, member.id)

    # Remove dict element on undeaf
    if was_undeafen:
        if DeafEvent.check_event(guild_id, member.id):
            DeafEvent.remove_event(guild_id, member.id)


# User updated status, activity, nickname or roles
@client.event
async def on_member_update(before: discord.Member, after: discord.Member):
    were_roles_changed: bool = before.roles is not after.roles

    # Abort amputate event if roles changed
    if were_roles_changed:
        is_amputated: bool = AmputateEvent.check_event(after.guild.id, after.id)
        if is_amputated:
            amputate_event: AmputateEvent = AmputateEvent.get_event(after.guild.id, after.id)
            happened_recently: bool = amputate_event.get_recent_state()
            if not happened_recently:
                AmputateEvent.remove_event(after.guild.id, after.id)


# Member join guild
@client.event
async def on_guild_join(guild: discord.Guild):
    was_bot_added: bool = not MuteEvent.check_guild(guild.id)

    # Add guild dict
    if was_bot_added:
        for restrict_event_class in RestrictEvent.__subclasses__():
            restrict_event_class.add_guild(guild.id)

        Video.add_queue(guild.id)
        Video.queues[guild.id] = {}
        Video.ids[guild.id] = 1
        Video.last_video_plays[guild.id] = 0.0
        Video.last_video_pauses[guild.id] = 0.0
        Video.last_video_remaining_durations[guild.id] = 0


# Member leave guild
@client.event
async def on_member_remove(member: discord.Member):
    was_bot_removed: bool = member.id == bot_id

    # Remove guild dict
    if was_bot_removed:
        for restrict_event_class in RestrictEvent.__subclasses__():
            restrict_event_class.remove_guild(member.guild.id)
        guild: discord.Guild = member.guild

        Video.remove_queue(guild.id)
        del Video.queues[guild.id]
        del Video.ids[guild.id]
        del Video.last_video_plays[guild.id]
        del Video.last_video_pauses[guild.id]
        del Video.last_video_remaining_durations[guild.id]


client.run(token)
