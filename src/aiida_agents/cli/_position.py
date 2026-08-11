"""Let the root group's options be written on either side of the subcommand.

Click decides whether a flag belongs to the group or to the command by its
*position*: ``--agent`` is declared on the root group, so Click stops looking
for it the moment it reads a subcommand name. That makes the obvious spelling
wrong::

    aiida-agents -a codegen ask "..."     # works
    aiida-agents ask -a codegen "..."     # No such option: -a

Every CLI a researcher uses daily --- git, docker, pip --- takes its flags
after the subcommand, so the second spelling is the one people reach for. And
Click's error names the option without mentioning that it exists, where it
goes, or that moving it two words to the left would fix it. It cost a
maintainer several minutes during dogfooding, which is generous: he had
written half of this CLI (`#79
<https://github.com/aiidateam/aiida-agents/issues/79>`_).

Two mechanisms here, because neither covers the whole surface alone.

:func:`accepts_root_options` puts the four options on a subcommand as well, so
the natural spelling simply works. It is applied where these flags mean
something --- ``ask`` and ``chat`` --- rather than everywhere, because
``--agent`` on ``rag build`` would be noise in the help output and a lie in the
parser.

:class:`PositionAwareGroup` catches what is left. A misplaced root option on
any other subcommand still fails, but now with a message naming the option and
showing where it goes, instead of a bare "No such option".
"""

from __future__ import annotations

import typing as t

import rich_click as click

__all__ = ["PositionAwareGroup", "accepts_root_options", "root_option_names"]

#: The root group's options, and the ``ctx.obj`` key each one fills.
#:
#: Read by both mechanisms, so a fifth root option cannot be added and be
#: accepted in one position but explained in the other.
_ROOT_OPTIONS: dict[str, tuple[str, ...]] = {
    "provider": ("--provider",),
    "model": ("--model",),
    "profile": ("--profile",),
    "agent": ("--agent", "-a"),
}


def root_option_names() -> frozenset[str]:
    """Every spelling of a root option, long and short."""
    return frozenset(flag for flags in _ROOT_OPTIONS.values() for flag in flags)


def _store(key: str) -> t.Callable[[click.Context, click.Parameter, t.Any], t.Any]:
    """Write a subcommand-side value into ``ctx.obj``, if one was given.

    ``None`` means the flag was absent here, and absent must not overwrite what
    the root group already stored --- otherwise the subcommand's *default*
    would silently beat an explicit ``aiida-agents -a codegen ask``. This is
    why the duplicated options carry no defaults of their own: it is the only
    way to tell "not given" from "given the default".

    When both sides carry a value the subcommand wins, being the more specific
    of the two, and the later of the two to be parsed.
    """

    def callback(ctx: click.Context, _param: click.Parameter, value: t.Any) -> t.Any:
        if value is not None:
            ctx.ensure_object(dict)
            ctx.obj[key] = value.lower() if key == "agent" else value
        return value

    return callback


def accepts_root_options(
    command: t.Callable[..., t.Any],
) -> t.Callable[..., t.Any]:
    """Also accept the root group's options after this subcommand's name.

    The values are folded into ``ctx.obj`` by callback rather than passed as
    arguments, so the decorated function's signature is unchanged and the two
    positions cannot drift into meaning different things.
    """
    from aiida_agents._settings import _PROVIDER_CHOICES
    from aiida_agents.cli.agent import _AGENT_CHOICES

    # Applied in reverse so `--help` lists them in the same order as the root
    # group does.
    for option in (
        click.option(
            "--agent",
            "-a",
            "agent_",
            type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
            default=None,
            expose_value=False,
            callback=_store("agent"),
            help="Which agent runs this request. Also accepted before the command.",
        ),
        click.option(
            "--profile",
            "profile_",
            default=None,
            expose_value=False,
            callback=_store("profile"),
            help="AiiDA profile to load. Also accepted before the command.",
        ),
        click.option(
            "--model",
            "model_",
            default=None,
            expose_value=False,
            callback=_store("model"),
            help="Override the model name. Also accepted before the command.",
        ),
        click.option(
            "--provider",
            "provider_",
            type=click.Choice(_PROVIDER_CHOICES, case_sensitive=False),
            default=None,
            expose_value=False,
            callback=_store("provider"),
            help="Override the model provider. Also accepted before the command.",
        ),
    ):
        command = option(command)
    return command


class PositionAwareGroup(click.RichGroup):
    """A root group whose misplaced-option error says where the option goes.

    For subcommands that do not take the root options themselves, this turns::

        Error: No such option: -a

    into a message naming the option and showing the working spelling. The
    parse still fails --- accepting ``--agent`` on ``rag build`` would be
    pretending it does something --- but the user is told the one thing they
    need, rather than left to guess that a flag they read in ``--help`` is
    position-sensitive.
    """

    def invoke(self, ctx: click.Context) -> t.Any:
        try:
            return super().invoke(ctx)
        except click.NoSuchOption as exc:
            if exc.option_name not in root_option_names():
                raise
            subcommand = ctx.invoked_subcommand or "<command>"
            message = (
                f"{exc.option_name} is an option of `aiida-agents` itself, so "
                f"it goes before the command:\n\n"
                f"    aiida-agents {exc.option_name} ... {subcommand} ...\n\n"
                f"(`ask` and `chat` accept it in either position.)"
            )
            raise click.UsageError(message, ctx=exc.ctx) from exc
