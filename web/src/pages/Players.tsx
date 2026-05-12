import { useEffect, useState } from "react";
import { api, errorMessage } from "../api";
import { AgentField } from "../components/AgentField";
import {
  FormBox,
  FormFooter,
  FormRow,
  PrimaryButton,
  SecondaryButton,
  Select,
  TextInput,
} from "../components/Form";
import { Notice } from "../components/Notice";
import { LoadingState, PageHeader, SectionHeader } from "../components/Page";
import { Td, Th } from "../components/Tables";
import { formatAgent } from "../format";
import type { AgentSpec, CheckpointInfo, PlayerRecord } from "../types";

export function PlayersPage() {
  const [players, setPlayers] = useState<PlayerRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch helper kept separate so the create/delete handlers can refresh
  // explicitly. The initial mount uses its own cancellation-aware effect.
  const refresh = () => {
    api
      .listPlayers()
      .then(setPlayers)
      .catch((e) => setError(errorMessage(e)));
  };

  useEffect(() => {
    let cancelled = false;
    api
      .listPlayers()
      .then((ps) => {
        if (!cancelled) setPlayers(ps);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const onDelete = async (p: PlayerRecord) => {
    if (!confirm(`Delete player ${p.label}?`)) return;
    try {
      await api.deletePlayer(p.id);
      refresh();
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  return (
    <div>
      <PageHeader
        title="Players"
        description="Named identities for agents. Defaults are the classical baselines; custom players let you fix seeds, set search depth, and (later) save trained checkpoints."
      />

      <SectionHeader title="Add custom player" />
      <NewPlayerForm
        onCreated={() => {
          setError(null);
          refresh();
        }}
        onError={setError}
      />

      {error && <div className="mt-4"><Notice kind="error">{error}</Notice></div>}

      <div className="mt-8">
        <SectionHeader title="All players" />
        {!players ? (
          <LoadingState>Loading players…</LoadingState>
        ) : (
          <div className="border border-neutral-200 bg-white">
            <table className="w-full text-sm">
              <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
                <tr>
                  <Th>label</Th>
                  <Th>agent</Th>
                  <Th>kind</Th>
                  <Th>id</Th>
                  <Th align="right">{""}</Th>
                </tr>
              </thead>
              <tbody>
                {players.map((p) => (
                  <tr
                    key={p.id}
                    className="border-b border-neutral-100 last:border-0"
                  >
                    <Td>
                      <span className="text-neutral-900">{p.label}</span>
                      {p.is_default && (
                        <span className="ml-2 text-[10px] uppercase tracking-wide text-neutral-400 border border-neutral-200 px-1.5 py-0.5">
                          default
                        </span>
                      )}
                    </Td>
                    <Td className="font-mono text-xs text-neutral-600">
                      {formatAgent(p.agent_spec)}
                    </Td>
                    <Td className="text-xs text-neutral-500">{p.agent_spec.kind}</Td>
                    <Td className="font-mono text-xs text-neutral-400">{p.id}</Td>
                    <Td align="right">
                      {!p.is_default && (
                        <SecondaryButton size="sm" onClick={() => onDelete(p)}>
                          delete
                        </SecondaryButton>
                      )}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function NewPlayerForm({
  onCreated,
  onError,
}: {
  onCreated: () => void;
  onError: (msg: string) => void;
}) {
  const [label, setLabel] = useState("");
  const [agent, setAgent] = useState<AgentSpec>({ kind: "greedy", seed: 1 });
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[] | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .listCheckpoints()
      .then((items) => {
        if (!cancelled) setCheckpoints(items);
      })
      .catch(() => {
        if (!cancelled) setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const submit = async () => {
    if (!label.trim()) return;
    setBusy(true);
    try {
      await api.createPlayer({ label: label.trim(), agent_spec: agent });
      setLabel("");
      onCreated();
    } catch (e) {
      onError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <FormBox>
      <FormRow label="Label">
        <TextInput value={label} onChange={setLabel} placeholder="Greedy seed=7" />
      </FormRow>
      <FormRow label="Agent">
        <AgentField value={agent} onChange={setAgent} allowHuman={false} />
      </FormRow>
      {agent.kind === "alphazero" && checkpoints && checkpoints.length > 0 && (
        <FormRow label="Checkpoint">
          <Select
            value={agent.checkpoint_path}
            options={[
              { value: "", label: "choose checkpoint" },
              ...checkpoints.map((c) => ({ value: c.path, label: c.label })),
            ]}
            width="w-96"
            onChange={(path) => {
              setAgent({ ...agent, checkpoint_path: path });
              const ckpt = checkpoints.find((c) => c.path === path);
              if (ckpt && !label.trim()) setLabel(`AlphaZero ${ckpt.label}`);
            }}
          />
        </FormRow>
      )}
      <FormFooter>
        <span className="text-xs text-neutral-500">
          Defaults are seeded automatically. Add a player for any agent + parameter
          combination you'll want to refer to in analyses and tournaments.
        </span>
        <PrimaryButton onClick={submit} disabled={busy || !label.trim()}>
          {busy ? "Adding..." : "Add player"}
        </PrimaryButton>
      </FormFooter>
    </FormBox>
  );
}
