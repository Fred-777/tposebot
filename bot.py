# Safe class annotations
from __future__ import annotations
# General types
from typing import *
# Abstract classes and methods
from abc import ABC, abstractmethod
# Everything from discord
from discord import *

import aiohttp
import asyncio
import bs4
import datetime
import dotenv
import json
import librosa
import logging
import os
import random
import re
import sys
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

extra_srcs_path: str = "./extra-srcs.txt"

youtube_videos_source_dir: str = "youtube-videos"
cursed_audios_dir: str = "cursed-audios"
youtube_videos_download_dir: str = "youtube-videos-download"
youtube_base_url: str = "https://www.youtube.com"

n_words_path: str = "./n-words.txt"

# ---------- Debugging ---------- #

logging.basicConfig(level=logging.INFO, filename="./log.txt")
logger = logging.getLogger("Discord")
handler = logging.FileHandler(filename='./log.txt', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)


# ---------- Text formatting ---------- #


def format_bold_text(text: str) -> str:
	""" Format text to become bold. """
	return f"**{text}**"


def format_code_block(text: str) -> str:
	""" Format text to become multiline code block. """
	return f"```{text}```"


# ---------- Load environment ---------- #

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

# Remove incomplete downloads
for filename in os.listdir(f"./{youtube_videos_download_dir}"):
	os.remove(f"./{youtube_videos_download_dir}/{filename}")

# Prevent cache from ruining play command
os.system("youtube-dl --rm-cache-dir")

with open(extra_srcs_path) as file:
	data: str = file.read()
	extra_srcs: List[str] = json.loads(data)


# ---------- Class helpers ---------- #


def suggest_help() -> str:
	""" Simple introduction to be performed when bot is highlighted. """
	return f"Type {prefix}help for more info"


author_id: int = 239388097714978817
bot_id: int = 647954736959717416
game: Game = Game(suggest_help())
intents: Intents = Intents.all()
client: Client = Client(activity=game, intents=intents)

# Referenced before connection
author_user: User = None
connect_datetime: datetime.datetime = None
awake_timeouts: Dict[int, Set[int]] = {}


# ---------- Classes ---------- #

class Command:
	""" Describe a command available from bot. """

	def __init__(self, name: str, description: str, example: str):
		self.name = name
		self.description = description
		self.example = example

	def get_brief_data(self) -> str:
		""" Get description for command. """
		return f"{self.name}: {self.description}"

	def get_extended_data(self) -> str:
		""" Get description and example for command. """
		return (f"Description: \n{self.description}\n\n" +
				f"Example: {self.example}")

	def __repr__(self):
		return self.name


class RestrictEvent(ABC):
	""" Base class for member restriction event management. """
	ids: Dict[int, int] = None
	events: Dict[int, Dict[int, AmputateEvent]] = None

	def __init__(self, member: Member, seconds: int):
		self.id = self.__class__.ids[member.guild.id]
		self.member: Member = member
		self.seconds: int = seconds
		self.start_time: float = time.time()

		self.__class__.ids[member.guild.id] += 1

	def get_remaining_seconds(self) -> int:
		""" Calculate amount of remaining seconds to end restriction. """
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
		""" Add guild as a instance container. """
		cls.ids[guild_id] = 1
		cls.events[guild_id] = {}

	@classmethod
	def remove_guild(cls, guild_id: int) -> None:
		""" Remove guild as a instance container. """
		del cls.ids[guild_id]
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
	ids: Dict[int, int] = {}
	events: Dict[int, Dict[int, AmputateEvent]] = {}

	name_present: str = "amputate"
	name_past: str = "amputated"
	function_role: Callable = lambda role: role.permissions.send_messages
	kwarg_key: str = "roles"

	def __init__(self, guild: Guild, member: Member, seconds: int):
		super().__init__(member, seconds)
		AmputateEvent.add_event(guild.id, self)

		self.previous_roles: List[Role] = self.member.roles

	def get_kwarg_enable(self) -> Dict[str, List[Role]]:
		""" Enable restriction and return required enable kwarg. """
		return {self.kwarg_key: []}

	def get_kwarg_disable(self) -> Dict[str, List[Role]]:
		""" Disable restriction and return required disable kwarg. """
		return {self.kwarg_key: self.previous_roles}


class DeafEvent(RestrictEvent):
	""" Deaf event on progress. """
	ids: Dict[int, int] = {}
	events: Dict[int, Dict[int, DeafEvent]] = {}

	name_present: str = "deaf"
	name_past: str = "deafened"
	function_role: Callable = lambda role: role.permissions.deafen_members
	kwarg_key: str = "deafen"

	def __init__(self, guild: Guild, member: Member, seconds: int):
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
	ids: Dict[int, int] = {}
	events: Dict[int, Dict[int, MuteEvent]] = {}

	name_present: str = "mute"
	name_past: str = "muted"
	function_role: Callable = lambda role: role.permissions.mute_members
	kwarg_key: str = "mute"

	def __init__(self, guild: Guild, member: Member, seconds: int):
		super().__init__(member, seconds)
		MuteEvent.add_event(guild.id, self)

	def get_kwarg_enable(self) -> Dict[str, bool]:
		""" Enable restriction and return required enable kwarg. """
		return {self.kwarg_key: True}

	def get_kwarg_disable(self) -> Dict[str, bool]:
		""" Disable restriction and return required disable kwarg. """
		return {self.kwarg_key: False}


class InactivityEvent:
	""" Represent a voice channel inactivity that may result in automatic leaving. """
	ids: Dict[int, int] = {}
	events: Dict[int, InactivityEvent] = {}

	def __init__(self, guild_id: int):
		self.id: int = InactivityEvent.ids[guild_id]
		self.guild_id: int = guild_id

		InactivityEvent.add_event(self)
		InactivityEvent.ids[guild_id] += 1

	@staticmethod
	def add_guild(guild_id: int) -> None:
		""" Add guild to guild dict. """
		InactivityEvent.ids[guild_id] = 1
		InactivityEvent.events[guild_id] = None

	@staticmethod
	def remove_guild(guild_id: int) -> None:
		""" Remove guild from guild dict. """
		del InactivityEvent.ids[guild_id]
		del InactivityEvent.events[guild_id]

	@staticmethod
	def add_event(inactivity_event: InactivityEvent) -> None:
		""" Add event to event slot. """
		InactivityEvent.events[inactivity_event.guild_id] = inactivity_event

	@staticmethod
	def remove_event_if_exists(guild_id: int) -> None:
		""" Remove event from event slot if it exists. """
		if guild_id in InactivityEvent.events:
			InactivityEvent.events[guild_id] = None

	@staticmethod
	def get_event(guild_id: int) -> InactivityEvent:
		""" Get current inactivity event for given guild. """
		return InactivityEvent.events[guild_id]

	@staticmethod
	def check_event(guild_id: int) -> bool:
		""" Inform if bot is inactive on given guild. """
		return InactivityEvent.events[guild_id] is not None


class Video:
	""" Represent a video to be played. """
	ids: Dict[int, int] = {}
	queues: Dict[int, Dict[int, Video]] = {}
	loops: Dict[int, bool] = {}
	last_video_file_paths: Dict[int, str] = {}

	# Text channel that triggered bot voice session in each guild
	caller_text_channels: Dict[int, TextChannel] = {}

	# Voice channel that bot is currently in in each guild
	connected_voice_channels: Dict[int, VoiceChannel] = {}

	# Last play/unpause timestamp in each guild
	last_video_plays: Dict[int, float] = {}
	# Last pause timestamp in each guild
	last_video_pauses: Dict[int, float] = {}
	# Last play/unpause remaining duration in each guild
	last_video_remaining_durations: Dict[int, int] = {}

	def __init__(self, title: str, duration: int, message: Message, file_path: str):
		for char_code in big_char_codes:
			char: str = chr(char_code)
			title = title.replace(char, " ")
		partial_title: str = title if len(title) <= 40 else f"{title[: 37]}..."

		guild: Guild = message.guild
		self.id: int = Video.ids[guild.id]
		self.partial_title: str = partial_title
		self.title: str = title
		self.duration: int = duration
		self.guild: Guild = guild
		self.channel: TextChannel = message.channel
		self.file_path = file_path
		self.audio_source: FFmpegPCMAudio = FFmpegPCMAudio(file_path)

		Video.ids[guild.id] += 1
		Video.add_video(guild.id, self)

	def get_formatted_duration(self) -> str:
		""" Get formatted representation of duration. """
		return get_formatted_duration(self.duration, justify=True)

	@classmethod
	def get_remaining_duration(cls, guild: Guild) -> int:
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
	def get_formatted_remaining_duration(cls, guild: Guild) -> str:
		""" Get formatted representation of remaining duration for current video. """
		remaining_duration: int = Video.get_remaining_duration(guild)
		return get_formatted_duration(remaining_duration)

	@classmethod
	def get_total_formatted_remaining_duration(cls, guild: Guild) -> str:
		""" Get formatted representation of remaining duration for videos in queue. """
		current_remaining_duration: int = Video.get_remaining_duration(guild)

		videos: List[Video] = Video.get_videos(guild.id)
		next_videos: List[Video] = videos[1:]
		next_videos_duration: Generator[int] = (video.duration for video in next_videos)
		total_next_videos_duration: int = sum(next_videos_duration)

		total_remaining_duration: int = current_remaining_duration + total_next_videos_duration

		return get_formatted_duration(total_remaining_duration)

	@classmethod
	def add_queue(cls, guild_id: int) -> None:
		""" Add video dict to guild dict. """
		cls.ids[guild_id] = 1
		cls.queues[guild_id] = {}
		cls.loops[guild_id] = False
		cls.last_video_file_paths[guild_id] = None
		cls.caller_text_channels[guild_id] = None
		cls.connected_voice_channels[guild_id] = None
		cls.last_video_plays[guild_id] = 0.0
		cls.last_video_pauses[guild_id] = 0.0
		cls.last_video_remaining_durations[guild_id] = 0

	@classmethod
	def remove_queue(cls, guild_id: int) -> None:
		""" Remove video dict from guild dict. """
		del cls.ids[guild_id]
		del cls.queues[guild_id]
		del cls.loops[guild_id]
		del cls.last_video_file_paths[guild_id]
		del cls.caller_text_channels[guild_id]
		del cls.connected_voice_channels[guild_id]
		del cls.last_video_plays[guild_id]
		del cls.last_video_pauses[guild_id]
		del cls.last_video_remaining_durations[guild_id]

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
	def get_formatted_queue(cls, guild: Guild) -> str:
		""" Get formatted queue of videos in guild queue. """
		videos: List[Video] = cls.get_videos(guild.id)

		if len(videos) == 0:
			return "Queue is empty"

		voice_client: VoiceClient = guild.voice_client

		video_pluralized: str = pluralize(len(videos), "video", "videos")

		is_playing: bool = voice_client.is_playing()
		is_paused: bool = voice_client.is_paused()
		is_looping: bool = Video.loops[guild.id]

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
					   f"Current video loop: {'✓' if is_looping else '✗'}\n" +
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

		if len(videos_data) > 16:
			videos_data = [*videos_data[: 15], "." * len(table_header), videos_data[-1]]

		formatted_videos: str = "\n".join(videos_data)

		return format_code_block(header + table_header + table_separator + formatted_videos)


# ---------- Exceptions ---------- #

class InvalidIntException(Exception):
	""" Exception triggered when a expected integer is invalid. """

	def __init__(self, name: str, min_value: int, max_value: int):
		self.name: str = name
		self.min_value: int = min_value
		self.max_value: int = max_value


class InvalidArgumentException(Exception):
	""" Exception triggered when a command receives invalid argument for a parameter. """

	def __init__(self, message: str):
		self.message: str = message


class MissingParameterException(Exception):
	""" Exception triggered when a command doesn't receive a required parameter. """

	def __init__(self, parameter_name: str):
		self.parameter_name: str = parameter_name


class TooManyArgumentsException(Exception):
	""" Exception triggered when a command receives more arguments than expected. """
	pass


class NoPermissionException(Exception):
	""" Exception triggered when a user requests an action which he doesn't have permission to execute. """

	def __init__(self, user_mention: str, action_name: str):
		self.user_mention: str = user_mention
		self.action_name: str = action_name


class AuthorNotInVoiceChannelException(Exception):
	""" Exception triggered when a user requests an action which requires him to be in a voice channel. """
	pass


class UserNotInVoiceChannelException(Exception):
	""" Exception triggered when a command requires a user to be in voice channel but he's not. """
	pass


class UserNotHighlightedException(Exception):
	""" Exception triggered when a command requires a user to be highlighted but he's not. """
	pass


class VideoTooLargeException(Exception):
	""" Exception triggered when a given video duration is above the time threshold. """

	def __init__(self, video_title: str):
		self.video_title: str = video_title


class NoVideoFoundException(Exception):
	""" Exception triggered when no youtube videos are found in a search query. """
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


async def _request_image_srcs(url: str):
	""" Lower level function to request image srcs from a single page. """
	async with aiohttp.ClientSession() as session:
		async with session.get(url) as response:
			content: str = str(await response.read())

	soup: bs4.BeautifulSoup = bs4.BeautifulSoup(content, features="lxml")
	imgs: bs4.element.Tag = soup.select("div > a > img")
	srcs: List[str] = [img["src"] for img in imgs]

	return srcs


async def request_image_srcs(query_param: str, max_page: int = 1) -> List[str]:
	""" Return image srcs found until a given page from query service. """
	base_url: str = "http://results.dogpile.com"

	request_headers: Dict[str, str] = {
		"Accept": "text/html",
		"User-Agent": "Chrome"
	}

	urls: Generator[str] = (f"{base_url}/serp?qc=images&q={query_param}&page={page}&sc={sc_param}"
							for page in range(1, max_page + 1))

	tasks: List[asyncio.Task] = [event_loop.create_task(_request_image_srcs(url)) for url in urls]
	await asyncio.wait(tasks)
	pages_srcs: List[List[str]] = [task.result() for task in tasks]
	srcs: List[str] = [src for page_srcs in pages_srcs for src in page_srcs]

	return srcs


async def request_tpose_srcs(max_page: int = 1) -> List[str]:
	""" Request tpose image srcs. """
	srcs: List[str] = await request_image_srcs("tpose", max_page)
	srcs.extend(extra_srcs)

	return srcs


def check_permissions(author: Member, function_role: Callable) -> bool:
	""" Inform if author has permission to perform a given action. """
	roles: List[Role] = author.roles

	role_has_permission: bool = any([function_role(role) for role in roles])
	is_admin: bool = any([role.permissions.administrator for role in roles])
	is_owner: bool = author.id == author.guild.owner_id
	# is_creator: bool = author.id == author_id

	has_permission: bool = role_has_permission or is_admin or is_owner

	return has_permission


async def restrict(message: Message,
				   arguments: List[str],
				   restrict_event_class: ClassVar[RestrictEvent]) -> str:
	""" Restrict someone for given number of seconds. """
	length: int = len(arguments)
	required_length: int = 3

	# Validate length
	if length == 1:
		raise MissingParameterException("member")
	if length == 2:
		raise MissingParameterException("seconds")
	if length > required_length:
		raise TooManyArgumentsException()

	member_name: str = arguments[1]
	member_id: int = extract_id(member_name)
	seconds_str: str = arguments[2]

	# Validate member highlight
	is_highlight: bool = member_id != -1
	if not is_highlight:
		return "Target user must be highlighted"

	# Validate author permissions
	author: Member = message.author
	function_role: Callable = restrict_event_class.function_role
	has_permission: bool = check_permissions(author, function_role)

	if not has_permission:
		raise NoPermissionException(author.mention, restrict_event_class.name_present)

	# Get member
	members: List[Member] = [member for member in message.guild.members]
	member: Member = next((member for member in members if member.id == member_id), None)

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
	except errors.HTTPException as e:
		if restrict_event_class.name_present in voice_restrictions:
			return "Target user is not in voice chat"
		else:
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


async def unrestrict(message: Message,
					 arguments: List[str],
					 restrict_event_class: ClassVar[RestrictEvent]) -> str:
	""" Unrestrict a restricted member. """
	length: int = len(arguments)
	required_length: int = 2

	# Validate length
	if length == 1:
		raise MissingParameterException("member")
	if length > required_length:
		raise TooManyArgumentsException()

	member_name: str = arguments[1]
	member_id: int = extract_id(member_name)

	# Validate member highlight
	is_highlight: bool = member_id != -1
	if not is_highlight:
		return "Target user must be highlighted"

	# Validate author permissions
	author: Member = message.author
	function_role: Callable = restrict_event_class.function_role
	has_permission: bool = check_permissions(author, function_role)

	if not has_permission:
		raise NoPermissionException(author.mention, restrict_event_class.name_present)

	# Get member
	members: List[Member] = [member for member in message.guild.members]
	member: Member = next((member for member in members if member.id == member_id), None)

	# Unrestrict if still restricted
	guild_id: int = message.guild.id
	is_restricted: bool = restrict_event_class.check_event(guild_id, member.id)
	if is_restricted:
		restrict_event: RestrictEvent = restrict_event_class.get_event(guild_id, member.id)
		kwarg_disable: Dict[str, Any] = restrict_event.get_kwarg_disable()
		await member.edit(**kwarg_disable)
	else:
		return f"User {member.mention} is not time-{restrict_event_class.name_past}"


async def restrictionlist(message: Message,
						  arguments: List[str],
						  restrict_event_class: ClassVar[RestrictEvent]) -> str:
	""" Get list of currently restricted members on requested guild. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	return restrict_event_class.get_formatted_list(message.guild.id)


async def process_message(message: Message) -> str:
	""" Handle message reply. """
	reply: str = None

	# Verify if message is just bot highlight
	bot_was_highlighted: bool = re.search(f"^<@!?{bot_id}>$", message.content) is not None

	# Handle input
	arguments: List[str] = re.findall("\S+", message.content)
	command: str = arguments[0]
	is_help: bool = re.search(f"^{prefix}help", message.content) is not None
	command_exists: bool = command in commands_map
	is_private: bool = message.guild is None

	# Deal with private message
	if is_private:
		pass
	# Reply highlight
	elif bot_was_highlighted:
		reply = suggest_help()
	# Reply to help
	elif is_help:
		reply = get_help(arguments)
	# Run command
	elif command_exists:
		command_function: Callable = commands_map[command]
		print(f"[{message.guild}] {message.author}: {message.content}")
		reply = await command_function(message, arguments)

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


async def request_voice_client(message: Message) -> VoiceClient:
	""" Request voice client for author voice channel and connect bot if it's not connected. """
	author: Member = message.author
	guild: Guild = message.guild
	is_bot_in_channel: bool = guild.voice_client is not None
	if is_bot_in_channel:
		voice_client: VoiceClient = guild.voice_client
	else:
		voice_client: VoiceClient = await author.voice.channel.connect()

		# Add caller text channel
		Video.caller_text_channels[guild.id] = message.channel

	return voice_client


async def disconnect(guild: Guild) -> None:
	""" Disconnect voice client from guild. """
	await guild.voice_client.disconnect()
	Video.clear_queue(guild.id)


async def update_queue(guild: Guild, video: Video, is_skip: bool = False) -> None:
	""" Handle video add/play. """
	queue: Dict[int, Video] = Video.get_queue(guild.id)
	is_play: bool = len(queue) == 1
	is_looping: bool = Video.loops[guild.id]
	InactivityEvent.remove_event_if_exists(guild.id)

	try:
		if is_skip:
			if not is_looping:
				Video.remove_next_video(guild.id)
			guild.voice_client.stop()
		next_video: Video = Video.get_next_video(guild.id)

		# Update last video loop cache
		Video.last_video_file_paths[guild.id] = next_video.file_path

		if not guild.voice_client.is_playing():
			if is_looping:
				video_file_path: str = Video.last_video_file_paths[guild.id]
				next_video.audio_source = FFmpegPCMAudio(video_file_path)
			is_paused: bool = guild.voice_client.is_paused()
			if not is_paused:
				guild.voice_client.play(next_video.audio_source,
										after=lambda e: event_loop.create_task(update_queue(guild,
																							video,
																							is_skip=True)))

		# Update remaining duration on video play / skip and last play/pause timestamp
		if is_play or is_skip:
			Video.last_video_plays[guild.id] = time.time()
			Video.last_video_remaining_durations[guild.id] = next_video.duration

	except StopIteration as e:
		filenames: List[str] = os.listdir(f"./{youtube_videos_download_dir}")
		is_downloading: bool = len(filenames) > 0
		if not is_downloading:

			seconds_inactive: int = 300
			inactivity_event: InactivityEvent = InactivityEvent(guild.id)
			await asyncio.sleep(seconds_inactive)

			current_inactivity_event: InactivityEvent = InactivityEvent.get_event(guild.id)
			is_same_restriction: bool
			try:
				is_same_restriction = inactivity_event.id == current_inactivity_event.id
			except AttributeError:
				is_same_restriction = False

			is_inactive: bool = InactivityEvent.check_event(guild.id)
			if is_inactive and is_same_restriction:
				# Read caller text channel and connected voice channel
				text_channel: TextChannel = Video.caller_text_channels[guild.id]
				voice_channel: VoiceChannel = Video.connected_voice_channels[guild.id]
				Video.last_video_file_paths[guild.id] = None
				await disconnect(guild)
				await text_channel.send(f"I left {voice_channel.mention} for being inactive")


def shuffle_dict(d: Dict[int, Any]) -> Dict[int, Any]:
	""" Shuffle a given dict. """
	keys: List[int] = list(d.keys())
	random.shuffle(keys)
	shuffled_d: Dict[int, Any] = {key: d[key] for key in keys}

	return shuffled_d


def count_big_chars(s: str) -> int:
	""" Count big chars in a given string. """
	return sum(ord(char) in big_char_codes for char in s)


def get_current_datetime() -> datetime.datetime:
	""" Request current UTC time and get datetime object from it. """
	return datetime.datetime.fromtimestamp(time.time()) + datetime.timedelta(hours=3, minutes=0, seconds=0)


# return datetime.datetime.now()


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


async def update_queue_and_feedback(guild: guild, video: Video) -> str:
	""" Update queue and provide feedback about last video appended. """
	queue: Dict[int, Video] = Video.get_queue(guild.id)
	state: str = "Playing" if len(queue) == 1 else "Queued"
	await update_queue(guild, video)

	return f"{state} {video.title}"


def get_python_version() -> str:
	""" Get current python version. """
	version = re.search("\S+", sys.version)[0]

	return version


def get_seconds_difference(datetime1: datetime.datetime, datetime2: datetime.datetime) -> int:
	""" Get seconds difference between two given datetimes. """
	result_datetime: datetime.timedelta = datetime1 - datetime2
	total_seconds: float = result_datetime.total_seconds()
	seconds_difference: int = abs(int(total_seconds))

	return seconds_difference


async def add_youtube_video(url: str, message: Message) -> Video:
	""" Process given youtube video url, add it to queue and return it. """
	video_id: str = re.search("(?<=v=)[\w\-]{11}", url)[0]

	with youtube_dl.YoutubeDL() as ydl:
		video_info: Dict = await event_loop.run_in_executor(None,
															lambda: ydl.extract_info(url,
																					 download=False))

	video_title: str = video_info["title"]
	video_duration: int = int(video_info["duration"])
	max_seconds_amount: int = 7200

	if video_duration > max_seconds_amount:
		raise VideoTooLargeException(video_title)

	source_file_path: str = f"./{youtube_videos_source_dir}/{video_id}.mp3"
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
			await event_loop.run_in_executor(None,
											 lambda: ydl.download([url]))

		filenames: List[str] = os.listdir(youtube_videos_download_dir)
		filename: str = next(filename for filename in filenames
							 if filename.find(video_id) > -1)
		os.rename(f"./{youtube_videos_download_dir}/{filename}", source_file_path)

	video: Video = Video(video_title, video_duration, message, source_file_path)

	return video


async def request_playlist_video_ids(playlist_id: str) -> List[int]:
	""" Request video ids for a given playlist. """
	video_ids: List[int] = []
	max_results: int = 50
	page_token: str = ""
	response_obj: Dict = None

	has_next_page: bool = True
	while has_next_page:
		query_params = (f"part=snippet&playlistId={playlist_id}&key={api_key}&maxResults={max_results}&" +
						f"pageToken={page_token}")
		url: str = f"https://www.googleapis.com/youtube/v3/playlistItems?{query_params}"

		async with aiohttp.ClientSession() as session:
			async with session.get(url) as response:
				response_obj = await response.json()

		for item in response_obj["items"]:
			video_ids.append(item["snippet"]["resourceId"]["videoId"])

		has_next_page = "nextPageToken" in response_obj
		if has_next_page:
			page_token = response_obj["nextPageToken"]

	return video_ids


# ---------- Interface ---------- #


def get_help(arguments: List[str]) -> str:
	""" Describe all commands briefly or a given command extensively. """
	length: int = len(arguments)
	required_lengths: Set[int] = {1, 2}
	max_required_length: int = max(required_lengths)
	is_general_help: bool = length == 1
	is_command_help: bool = length == 2

	if length > max_required_length:
		raise TooManyArgumentsException()

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
			command: str = arguments[1]
			return commands[command].get_extended_data()
		except KeyError:
			return "Invalid command was given"


async def code(message: Message, arguments: List[str]) -> None:
	""" Provide file with current source code. """
	length: int = len(arguments)
	required_length: int = 1

	if length > required_length:
		raise TooManyArgumentsException()

	code_file: File = File("./stable-bot.py")
	await message.channel.send(file=code_file)


async def amputate(message: Message, arguments: List[str]) -> str:
	""" Amputate someone for given number of seconds. """
	return await restrict(message, arguments, AmputateEvent)


async def unamputate(message: Message, arguments: List[str]) -> str:
	""" Unamputate a amputated member. """
	return await unrestrict(message, arguments, AmputateEvent)


async def amputatelist(message: Message, arguments: List[str]) -> str:
	""" Get list of currently amputated members on requested guild. """
	return await restrictionlist(message, arguments, AmputateEvent)


async def deaf(message: Message, arguments: List[str]) -> str:
	""" Deafen someone for given number of seconds. """
	return await restrict(message, arguments, DeafEvent)


async def undeaf(message: Message, arguments: List[str]) -> str:
	""" Undeaf a deafened member. """
	return await unrestrict(message, arguments, DeafEvent)


async def deaflist(message: Message, arguments: List[str]) -> str:
	""" Get list of currently deafened members on requested guild. """
	return await restrictionlist(message, arguments, DeafEvent)


async def mute(message: Message, arguments: List[str]) -> str:
	""" Mute someone for given number of seconds. """
	return await restrict(message, arguments, MuteEvent)


async def unmute(message: Message, arguments: List[str]) -> str:
	""" Unmute a muted member. """
	return await unrestrict(message, arguments, MuteEvent)


async def mutelist(message: Message, arguments: List[str]) -> str:
	""" Get list of currently muted members on requested guild. """
	return await restrictionlist(message, arguments, MuteEvent)


async def serverlist(message: Message, arguments: List[str]) -> str:
	""" Request list of servers in which TPoseBot is present. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	guilds: List[Guild] = sorted([guild for guild in client.guilds if guild is not None],
								 key=lambda guild: guild.name.lower())
	header: str = f"{len(guilds)} servers found\n\n"

	name_lengths: Set[int] = {len(guild.name) for guild in guilds}
	highest_name_length: int = max(name_lengths)

	guilds_data: List[str] = [f"{guild.name.ljust(highest_name_length)} | " +
							  f"Member count: {len(guild.members)}"
							  for guild in guilds]
	formatted_guilds: str = "\n".join(guilds_data)

	return format_code_block(header + formatted_guilds)


async def tpose(message: Message, arguments: List[str]) -> str:
	""" Request random tpose image. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	src: str = random.choice(srcs)

	return src


async def cursed(message: Message, arguments: List[str]) -> str:
	""" Request bot to join voice channel and play a random cursed audio. """
	length: int = len(arguments)
	required_length: int = 1

	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

	filenames: List[str] = os.listdir(f"./{cursed_audios_dir}")
	filename: str = random.choice(filenames)
	file_path = f"./{cursed_audios_dir}/{filename}"

	duration: int = int(librosa.get_duration(filename=file_path))

	video: Video = Video(filename, duration, message, file_path)
	result: str = await update_queue_and_feedback(message.guild, video)

	return result


async def play(message: Message, arguments: List[str]) -> None:
	""" Request bot to join voice channel and optionally play a song. """
	length: int = len(arguments)

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

	has_query: bool = length > 1
	query: str = " ".join(arguments[1:]) if has_query else None

	# Search for something if query was given
	if query is not None:

		is_query: str = query.find(youtube_base_url) == -1
		youtube_video_url: str = None
		if is_query:
			try:
				youtube_video_url = await request_video_url(query)
			except IndexError:
				raise NoVideoFoundException()
		else:
			youtube_video_url = query

		query_params_regex: str = "(?<=\?).+"
		query_params_match: re.Match = re.search(query_params_regex, youtube_video_url)
		query_params: str = query_params_match[0] if query_params_match is not None else None

		# Build a key-value query param dict
		query_params_splitted: List[str] = query_params.split("&")
		query_params_key_valued: List[List[str]] = [s.split("=") for s in query_params_splitted]
		query_params_map: Dict[str, str] = {item[0]: item[1] for item in query_params_key_valued}

		# Handle query param values
		playlist_key: str = "list"
		has_playlist: bool = playlist_key in query_params_map
		video_key: str = "v"
		has_video: bool = video_key in query_params_map

		if has_playlist:
			# Play youtube playlist (basically list of youtube videos)
			playlist_id: str = query_params_map[playlist_key]
			video_ids: List[int] = await request_playlist_video_ids(playlist_id)
			for video_id in video_ids:
				try:
					youtube_video_url = f"{youtube_base_url}/watch?v={video_id}"
					video: Video = await add_youtube_video(youtube_video_url, message)
					result: str = await update_queue_and_feedback(message.guild, video)
					await message.channel.send(result, delete_after=30)
				except youtube_dl.utils.DownloadError:
					await message.channel.send(f"Couldn't download video {video_id}")
		elif has_video:
			try:
				# Play youtube video
				video: Video = await add_youtube_video(youtube_video_url, message)
				result: str = await update_queue_and_feedback(message.guild, video)
				await message.channel.send(result)
			except youtube_dl.utils.DownloadError:
				video_id = query_params_map[video_key]
				await message.channel.send(f"Couldn't download video {video_id}")


async def leave(message: Message, arguments: List[str]) -> str:
	""" Request bot to leave voice channel. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	guild: Guild = message.guild
	is_in_voice_channel: bool = guild.voice_client is not None

	if is_in_voice_channel:
		await disconnect(guild)
	else:
		return "I am not connected to a voice channel"


async def pause(message: Message, arguments: List[str]) -> str:
	""" Request bot to pause current video. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

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


async def unpause(message: Message, arguments: List[str]) -> str:
	""" Request bot to unpause current video. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

	if voice_client.is_playing():
		return "I am already unpaused"
	else:
		voice_client.resume()

		# Update last video play
		Video.last_video_plays[message.guild.id] = time.time()


async def stop(message: Message, arguments: List[str]) -> str:
	""" Request bot to stop playing and reset queue. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

	voice_client.stop()
	Video.clear_queue(message.guild.id)


async def skip(message: Message, arguments: List[str]) -> str:
	""" Request bot to skip current video. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

	if voice_client.is_playing():
		message.guild.voice_client.stop()
	else:
		return "There are no videos to skip"


async def remove(message: Message, arguments: List[str]) -> str:
	""" Request bot to remove video with given id. """
	length: int = len(arguments)
	required_length: int = 2

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	is_user_in_channel: bool = message.author.voice is not None
	if not is_user_in_channel:
		raise AuthorNotInVoiceChannelException()

	voice_client: VoiceClient = await request_voice_client(message)

	# Check if id is number
	video_id_str = arguments[1]
	try:
		video_id: int = int(video_id_str)
	except ValueError:
		return "Given video id is not a number"

	# Attempt to remove video by id
	try:
		current_video: Video = Video.get_next_video(message.guild.id)
		to_be_removed_video: Video = Video.get_video(message.guild.id, video_id)
		is_current_video: bool = current_video.id == video_id

		if is_current_video:
			message.guild.voice_client.stop()
		else:
			Video.remove_video(message.guild.id, video_id)
	except KeyError:
		return "There is no video with given id"

	return f"{to_be_removed_video.title} was removed from queue"


async def shuffle(message: Message, arguments: List[str]) -> str:
	""" Shuffle videos in queue. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	voice_client: VoiceClient = await request_voice_client(message)

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


async def queue(message: Message, arguments: List[str]) -> str:
	""" Get current video queue. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	formatted_queue: str = Video.get_formatted_queue(message.guild)

	return formatted_queue


async def roll(message: Message, arguments: List[str]) -> str:
	""" Run a roll procedure which supports fixed and random values. """
	length: int = len(arguments)

	# Validate length
	if length == 1:
		raise MissingParameterException("max number")

	# Handle roll data
	outputs: List[str] = []
	operators: Set[str] = {"+", "-", "*"}

	min_value_repeat: int = 1
	max_value_repeat: int = 5
	min_value_prefix: int = 1
	max_value_prefix: int = 10
	min_value_range: int = 1
	max_value_range: int = 1000000

	roll_data: str = "".join(arguments[1:]).strip()
	repeat_amount: int = 1

	# Optional repetition prefix
	if re.search("^\d*#", roll_data) is not None:
		if roll_data.startswith("#"):
			roll_data = "1" + roll_data

		repeat_amount = int(re.search("^\d+", roll_data)[0])
		after_repeat_index: int = roll_data.find("#") + 1
		roll_data = roll_data[after_repeat_index:]

		is_repeat_valid: bool = validate_int(str(repeat_amount), min_value_repeat, max_value_repeat)

		if not is_repeat_valid:
			raise InvalidIntException("repeat", min_value_repeat, max_value_repeat)

	any_operator_regex = f"[{''.join(operators)}]"
	roll_data = re.sub(f"({any_operator_regex})", " \\1 ", roll_data)
	roll_arguments: List[str] = re.split("\s+", roll_data)

	for repeat in range(repeat_amount):
		arguments_result: List[str] = []
		is_operator_expected: bool = False
		last_operator: str = "+"

		for argument in roll_arguments:

			# Empty filter
			if argument == "":
				continue

			# Operator
			if argument in operators:
				if not is_operator_expected:
					raise InvalidArgumentException(f"Expected a number, received an operator instead: {argument}")

				is_operator_expected = False
				last_operator = argument

				# Multiplication behaves differently, it multiplies by everything until now rather than last number
				is_multiplication: bool = argument == "*"
				if is_multiplication:
					arguments_result[0] = "(" + arguments_result[0]
					arguments_result[-1] += ")"

				arguments_result.append(argument)

			# Received unexpected number
			elif is_operator_expected:
				raise InvalidArgumentException(f"Expected an operator, received a number instead: {argument}")

			# Received *?d*
			elif re.search("^\d*d\d+$", argument) is not None:
				if argument.startswith("d"):
					argument = "1" + argument

				is_operator_expected = True

				prefix: int = int(re.search("^\d+", argument)[0])
				max_range: int = int(re.search("\d+$", argument)[0])

				is_prefix_valid: bool = validate_int(str(prefix), min_value_prefix, max_value_prefix)
				is_range_valid: bool = validate_int(str(max_range), min_value_range, max_value_range)

				if not is_prefix_valid:
					raise InvalidIntException("prefix", min_value_prefix, max_value_prefix)
				if not is_range_valid:
					raise InvalidIntException("max range", min_value_range, max_value_range)

				for i in range(prefix):
					random_num: int = random.randint(1, max_range)
					if i > 0:
						arguments_result.append(last_operator)
					arguments_result.append(str(random_num))

			# Received single number
			elif re.search("^\d+$", argument) is not None:
				is_operator_expected = True
				arguments_result.append(argument)

			# Received unexpected value
			else:
				raise InvalidArgumentException(f"Invalid argument found: {argument}")

		# Prepare output
		result_code: str = " ".join(arguments_result)
		result: int = eval(result_code)
		output: str = f"{result_code} = {result}"
		outputs.append(output)

	final_output: str = "\n".join(outputs)

	return final_output


async def wipe(message: Message, arguments: List[str]) -> str:
	""" Remove all messages sent within last given number of seconds. """
	length: int = len(arguments)
	required_length: int = 2

	# Validate length
	if length == 1:
		raise MissingParameterException("seconds")

	if length > required_length:
		raise TooManyArgumentsException()

	# Validate author permissions
	author: Member = message.author
	function_role: Callable = lambda role: role.permissions.manage_messages
	has_permission: bool = check_permissions(author, function_role)

	if not has_permission:
		action_name: str = "wipe"
		raise NoPermissionException(author.mention, action_name)

	# Validate number
	total_seconds_str: str = arguments[1]
	min_value: int = 1
	max_value: int = 7200
	is_max_num_valid: int = validate_int(total_seconds_str, min_value, max_value)

	if not is_max_num_valid:
		raise InvalidIntException("seconds", min_value, max_value)

	total_seconds: int = int(total_seconds_str)

	now: datetime.datetime = get_current_datetime()
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

	await message.channel.send(result, delete_after=30)


async def nword(message: Message, arguments: List[str]) -> str:
	""" Get a random word that starts with n. """
	length: int = len(arguments)
	required_length: int = 1

	if length > required_length:
		raise TooManyArgumentsException()

	n_word: str = random.choice(n_words)

	return n_word


async def info(message: Message, arguments: List[str]) -> str:
	""" Get some general info about this bot. """
	length: int = len(arguments)
	required_length: int = 1

	if length > required_length:
		raise TooManyArgumentsException()

	language_name: str = "Python"
	language_version: str = get_python_version()
	playing_server_count: int = sum(1 if guild.voice_client is not None else 0 for guild in client.guilds)

	current_datetime: datetime.datetime = get_current_datetime()
	seconds_difference: int = get_seconds_difference(connect_datetime, current_datetime)
	active_time: str = get_formatted_duration(seconds_difference)

	output: List[str] = [
		f"Author: {author_user}",
		f"Programming language: {language_name} v{language_version}",
		f"Time running: {active_time}",
		f"Member in {len(client.guilds)} servers, currently playing in {playing_server_count}"
	]

	formatted_output: str = "\n".join(output)

	return format_code_block(formatted_output)


async def awake(message: Message, arguments: List[str]) -> str:
	""" Continuously moves a given member from voice channel back and forth for a while. """
	length: int = len(arguments)
	required_length: int = 2

	# Validate length
	if length == 1:
		raise MissingParameterException("member")

	if length > required_length:
		raise TooManyArgumentsException()

	member_name: str = arguments[1]
	member_id: int = extract_id(member_name)

	# Require member highlight
	is_highlight: bool = member_id != -1
	if not is_highlight:
		raise UserNotHighlightedException()

	function_role: Callable = lambda role: role.permissions.move_members
	has_permission: bool = check_permissions(message.author, function_role)

	if not has_permission:
		raise NoPermissionException(message.author.mention, "move")

	# Require author to be in voice channel
	author: Member = message.author
	is_author_in_channel: bool = author.voice is not None
	if not is_author_in_channel:
		raise AuthorNotInVoiceChannelException()

	# Require target user to be in voice channel
	member: Member = utils.find(lambda member: member.id == member_id, message.guild.members)
	is_user_in_channel: bool = member.voice is not None
	if not is_user_in_channel:
		raise UserNotInVoiceChannelException()

	# Voice channels which user can be moved to and are empty
	voice_channels: List[VoiceChannel] = [voice_channel for voice_channel in message.guild.voice_channels
										  if member.permissions_in(voice_channel).connect and
										  voice_channel.members == []]

	available_voice_channel_exists: bool = voice_channels != []
	if not available_voice_channel_exists:
		return f"There are no available empty voice channels to move {member.mention}"

	current_voice_channel: VoiceChannel = member.voice.channel
	other_voice_channel: VoiceChannel = voice_channels[0]

	guild_awake_timeouts: Set[int] = awake_timeouts[message.guild.id]
	is_timeouted: bool = author.id in guild_awake_timeouts
	if is_timeouted:
		return "Wait a moment before awaking again"

	move_reason: str = "Awaking purposes"
	seconds_duration: int = 30
	await member.move_to(other_voice_channel, reason=move_reason)
	await member.move_to(current_voice_channel, reason=move_reason)

	await message.channel.send(f"Wake up {member.mention}!!!")
	guild_awake_timeouts.add(author.id)
	await asyncio.sleep(seconds_duration)
	guild_awake_timeouts.remove(author.id)


async def loop(message: Message, arguments: List[str]) -> str:
	""" Toggle loop value for current guild. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	current_loop_value: bool = Video.loops[message.guild.id]
	next_loop_value: bool = not current_loop_value

	Video.loops[message.guild.id] = next_loop_value
	return f"Loop value was changed to {next_loop_value}"


async def search(message: Message, arguments: List[str]) -> str:
	""" Search for the meaning of a given word in online dictionary. """
	length: int = len(arguments)
	required_length: int = 2

	# Validate length
	if length == 1:
		raise MissingParameterException("word")

	if length > required_length:
		raise TooManyArgumentsException()

	result: str = ""
	word: str = arguments[1]
	base_url: str = "https://www.dictionary.com/browse"
	url: str = f"{base_url}/{word}"

	async with aiohttp.ClientSession() as session:
		async with session.get(url) as response:
			if response.status == 404:
				return "Error 404: query not found"
			elif not 200 <= response.status <= 299:
				return "Something went wrong"

			html: str = str(await response.read())

	soup: bs4.BeautifulSoup = bs4.BeautifulSoup(html, features="lxml")

	div_query: str = ".css-1fj93w9"
	assert soup.select(div_query) != [], "Div changed class name (disgraça)"
	div: bs4.element.Tag = soup.select(div_query)[0]
	sections: List[bs4.element.Tag] = list(div.select("section"))[1:]

	for section in sections:
		header: bs4.element.Tag = section.select("h3")[0]
		result += f"{header.text.upper()}\n"

		for i, item in enumerate(section.select("div[value]")):
			result += f"{i + 1}- {item.text.strip()}\n"

		result += "\n"

	if len(result) > 2000:
		result = f"{result[:1997]}..."

	return result


async def invite(message: Message, arguments: List[str]) -> str:
	""" Return invite url. """
	length: int = len(arguments)
	required_length: int = 1

	# Validate length
	if length > required_length:
		raise TooManyArgumentsException()

	invite_url: str = "https://discord.com/oauth2/authorize?client_id=647954736959717416&permissions=1408363632&scope=bot"
	return invite_url


# ---------- Application variables ---------- #

# Async scheduler
event_loop: asyncio.ProactorEventLoop = asyncio.get_event_loop()

# Tpose image srcs
srcs: List[str] = event_loop.run_until_complete(request_tpose_srcs(max_page=3))

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
	"roll": Command(f"{prefix}roll",
					("Roll a dice which supports fixed values (ex: 1, 2) and random values (ex: 1d10, 2d6). " +
					 "Random value must be prepended with a d."),
					f"Roll 2 random numbers from 1 to 10 and add 5 to it\n{prefix}roll 2d10 + 5"),
	"wipe": Command(f"{prefix}wipe",
					"Remove all messages sent within last given number of seconds",
					f"Remove all messages sent within last 30 seconds\n{prefix}wipe 30"),
	"nword": Command(f"{prefix}nword",
					 "Get a random word that starts with n",
					 f"\n{prefix}nword"),
	"info": Command(f"{prefix}info",
					"Show some general information about me",
					f"\n{prefix}info"),
	"awake": Command(f"{prefix}awake",
					 "Continuously moves a given member from voice channel back and forth for a while",
					 f"Awake member Fred\n{prefix}awake @Fred"),
	"loop": Command(f"{prefix}loop",
					"Activate/deactivate loop value on this server",
					f"Change loop value\n{prefix}loop"),
	"search": Command(f"{prefix}search",
					  "Search meaning for a given word",
					  f"Search for the meaning of the word hat\n{prefix}search hat"),
	"invite": Command(f"{prefix}invite",
					  "Get my invite url",
					  f"{prefix}invite")
}

commands_map: Dict[str, Callable] = {
	f"{prefix}help": get_help,
	f"{prefix}code": code,
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
	f"{prefix}roll": roll,
	f"{prefix}wipe": wipe,
	f"{prefix}nword": nword,
	f"{prefix}info": info,
	f"{prefix}awake": awake,
	f"{prefix}loop": loop,
	f"{prefix}search": search,
	f"{prefix}invite": invite
}

voice_restrictions: Set[str] = {"deaf", "mute"}

# Specific messages to be replied
special_messages: Dict[str, str] = {
	"ok": "boomer",
	"oi": "oi",
	"que": "ijo",
	"q": "ijo",
	"perdi": "perdi",
	"zabloing": "floppa",
	"googas": "floppa",
	"caracal": "floppa",
	"floppa": "floppa",
}


# ---------- Event listeners ---------- #

# Bot connect
@client.event
async def on_connect():
	# Initialize data that requires connection
	for guild in client.guilds:

		for restrict_event_class in RestrictEvent.__subclasses__():
			restrict_event_class.add_guild(guild.id)

		Video.add_queue(guild.id)
		InactivityEvent.add_guild(guild.id)

		global awake_timeouts
		awake_timeouts[guild.id] = set()

	app_info: AppInfo = await client.application_info()
	global author_user
	global connect_datetime
	author_user = app_info.owner
	connect_datetime = get_current_datetime()


# Bot ready
@client.event
async def on_ready():
	print(f"TPoseBot is currently in {len(client.guilds)} servers")

	max_guild_number_length: int = len(str(len(client.guilds)))
	max_guild_name_length: int = max(len(guild.name) for guild in client.guilds)
	for i, guild in enumerate(sorted(client.guilds, key=lambda guild: guild.name)):
		formatted_guild_number: str = str(i + 1).rjust(max_guild_number_length)
		formatted_guild_name: str = guild.name.ljust(max_guild_name_length)
		member_count: int = len(guild.members)
		print(f"{formatted_guild_number}: {formatted_guild_name} | Member count: {member_count}")

	# guild = next(guild for guild in client.guilds if "𝐓" in guild.name)
	# for channel in guild.channels:
	# 	try:
	# 		invite_url = await channel.create_invite(reason="a")
	# 		print(invite_url)
	# 		break
	# 	except Exception as e:
	# 		print(f"Zuou chat {channel.name} | {e}")

	content = guild.default_role


# guild_id = 376442713341558808
# guild = [guild for guild in client.guilds if guild.id == guild_id][0]
# text_channels = []
# channel_names = [f"spam{i + 1}" for i in range(25)]
# for channel_name in channel_names:
# 	try:
# 		channel = [channel for channel in guild.text_channels if channel.name == channel_name][0]
# 		await channel.delete(reason="Sou doente")
# 	except IndexError:
# 		pass

# for channel_name in channel_names:
# 	text_channel = await guild.create_text_channel(channel_name)
# 	text_channels.append(text_channel)
#
# while True:
# 	for text_channel in text_channels:
# 		await text_channel.send(content)


# Send message
@client.event
async def on_message(message: Message):
	sent_by_bot: bool = message.author.bot
	has_content: bool = message.content != ""

	if not sent_by_bot and has_content:
		try:
			channel: TextChannel = message.channel

			reply: str = await process_message(message)
			if reply is not None:
				await channel.send(reply)
		except InvalidIntException as e:
			await channel.send(f"Invalid {e.name}, it has to be an integer between {e.min_value} and {e.max_value}")
		except InvalidArgumentException as e:
			await channel.send(e.message)
		except MissingParameterException as e:
			await channel.send(f"No {e.parameter_name} parameter was given")
		except TooManyArgumentsException as e:
			await channel.send("Too many arguments were given")
		except NoPermissionException as e:
			await channel.send(f"User {e.user_mention} doesn't have permission to {e.action_name}")
		except UserNotHighlightedException as e:
			await channel.send(f"Given user must be highlighted")
		except AuthorNotInVoiceChannelException as e:
			await channel.send("You must be connected to a voice channel to use this command")
		except UserNotInVoiceChannelException as e:
			await channel.send("Given user must be connected to a voice channel to use this command")
		except VideoTooLargeException as e:
			await channel.send(f"Video {e.video_title} is too large")
		except NoVideoFoundException as e:
			await channel.send("No youtube videos were found for this search query")
		except UnicodeEncodeError:
			pass
		except youtube_dl.utils.DownloadError:
			await channel.send("Something went wrong with youtube-dl")

		# START CRINGE SECTION (non-official low effort features 4fun for specific guilds. There are no rules)

		# Allowed guild ids
		decente_guild_id: int = 289874563230072846
		elias_guild_id: int = 596731443741458452
		titas_id: int = 591648388399890450
		tcho_id: int = 649129370442530826
		griu_guild_id: int = 678343441750556672
		vixx_guild_id: int = 381298169385975809
		edu_coisa_id: int = 651515052280512533
		nao_sei_ids: List[int] = [693174750297587793, 376442713341558808, 479476923404124170, 707937125118640188,
								  677281522998575126, 740417342244388884, 757469095569522708, 699619134119477279]

		if message.guild is not None and message.guild.id in [titas_id, elias_guild_id, tcho_id, griu_guild_id,
									 # decente_guild_id,
									 vixx_guild_id,
									 edu_coisa_id,
									 *nao_sei_ids]:

			# Handle special messages
			content_lower: str = message.content.lower()
			key: str = next((key for key in special_messages
							 if re.search(f"^{key}(\s|$)", content_lower) is not None), None)
			is_special: bool = key is not None

			if is_special:
				special_message: str = special_messages[key]
				await message.channel.send(special_message)

			if message.content == "--apocalipse":

				voice_channels: List[VoiceChannel] = message.guild.voice_channels
				voice_members: List[Member] = [member for member in message.guild.members if member.voice is not None]

				seconds: int = 2
				start_time: float = time.time()

				while True:
					for voice_member in voice_members:
						other_voice_channels: List[VoiceChannel] = [channel for channel in voice_channels
																	if channel != voice_member.voice.channel]
						voice_channel: VoiceChannel = random.choice(other_voice_channels)
						await voice_member.move_to(voice_channel)

					if time.time() - start_time >= seconds:
						break

			if message.content == "--goiaba":

				author: Member = message.author
				voice_channel: VoiceChannel = message.author.voice.channel

				# Explosive goiaba audio
				voice_client: VoiceClient = await request_voice_client(message)
				file_path: str = f"{base_path}/ruim.mp3"
				video = Video(file_path, 99999, message, file_path)

				# Mute everyone from voice channel except the person who called it
				for member in voice_channel.members:
					if member.id not in [author.id, bot_id]:
						await member.edit(mute=True)

				await update_queue(message.guild, video)
				await message.channel.send("XDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")

			if message.content.lower() == "nossa":
				# Explosive goiaba audio
				voice_client: VoiceClient = await request_voice_client(message)
				file_path: str = f"{base_path}/nossa-q-bosta.mp3"
				video = Video(file_path, 99999999, message, file_path)

				await update_queue(message.guild, video)

			if message.content.lower() == "--tchau":

				voice_client: VoiceClient = await request_voice_client(message)
				file_path: str = f"{base_path}/tchau.mp3"
				video = Video(file_path, -3600, message, file_path)

				if voice_client.is_playing():
					message.guild.voice_client.stop()

				async def inner_disconnect(message: Message):
					voice_channel: VoiceChannel = message.author.voice.channel
					members: List[Member] = voice_channel.members

					await message.channel.send("Tchau")
					for member in members:
						await member.move_to(None)

				message.guild.voice_client.play(video.audio_source,
												after=lambda e: event_loop.create_task(inner_disconnect(message)))

			if message.content.startswith("--spam"):
				channel: DMChannel = await message.author.create_dm()
				await channel.send("Get spammed: 1 to 100")
				for i in range(1, 101):
					await channel.send(f"oi{i}")
					await asyncio.sleep(1)

			if message.content.startswith("--nf"):
				await message.delete()
				filename: str = random.choice(os.listdir("nf"))
				file: File = File(f"{base_path}/nf/{filename}")
				content: str = message.content[5:]
				await message.channel.send(content, file=file)


# END CRINGE SECTION


# Join, leave, mute, deafen on VC
@client.event
async def on_voice_state_update(member: Member, before: VoiceState, after: VoiceState):
	guild: Guild = member.guild
	was_unmuted: bool = before.mute and not after.mute
	was_undeafen: bool = before.deaf and not after.deaf
	has_joined: bool = before.channel is not after.channel and after.channel is not None
	has_left: bool = before.channel is not None and before.channel is not after.channel
	is_member_bot: bool = member.id == bot_id

	# Remove dict element on unmute
	if was_unmuted:
		if MuteEvent.check_event(guild.id, member.id):
			MuteEvent.remove_event(guild.id, member.id)

	# Remove dict element on undeaf
	elif was_undeafen:
		if DeafEvent.check_event(guild.id, member.id):
			DeafEvent.remove_event(guild.id, member.id)

	# Deal with member voice channel join
	if has_joined and is_member_bot:
		if is_member_bot:
			# Add / update connected voice channel
			Video.connected_voice_channels[guild.id] = guild.voice_client.channel

	# Deal with member voice channel leave
	if has_left:
		if is_member_bot:
			if not has_joined:
				Video.clear_queue(guild.id)
				Video.connected_voice_channels[guild.id] = None
		else:
			is_bot_in_channel: bool = bot_id in [member.id for member in before.channel.members]
			if is_bot_in_channel:
				is_bot_alone: bool = len(before.channel.members) == 1
				if is_bot_alone:
					# Read caller text channel and connected voice channel
					text_channel: TextChannel = Video.caller_text_channels[guild.id]
					voice_channel: VoiceChannel = Video.connected_voice_channels[guild.id]
					await disconnect(guild)
					await text_channel.send(f"I left {voice_channel.mention} because I was left alone")


# User updated status, activity, nickname or roles
@client.event
async def on_member_update(before: Member, after: Member):
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
async def on_guild_join(guild: Guild):
	print(f"Added to guild {guild.name}")
	was_bot_added: bool = not MuteEvent.check_guild(guild.id)

	# Add guild dict
	if was_bot_added:
		for restrict_event_class in RestrictEvent.__subclasses__():
			restrict_event_class.add_guild(guild.id)

		Video.add_queue(guild.id)

		InactivityEvent.add_guild(guild.id)

		global awake_timeouts
		awake_timeouts[guild.id] = set()


# Member leave guild
@client.event
async def on_member_remove(member: Member):
	was_bot_removed: bool = member.id == bot_id

	# Remove guild dict
	if was_bot_removed:
		print(f"Removed from guild {member.guild.name}")

		guild: Guild = member.guild

		for restrict_event_class in RestrictEvent.__subclasses__():
			restrict_event_class.remove_guild(guild.id)

		Video.remove_queue(guild.id)

		InactivityEvent.remove_guild(guild.id)

		global awake_timeouts
		del awake_timeouts[guild.id]


def main():
	client.run(token)


if __name__ == "__main__":
	main()
