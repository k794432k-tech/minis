//
//  AIChatViewModel+ModelPersona.swift
//  Minis — per-model soul, personality and memory in regular chats
//
//  When the chat's current model has an enabled persona (Settings → Model
//  Personas), its name, soul and personality replace the global SOUL.md
//  identity, its memories are added to the system prompt, and it gets the
//  `model_memory_write` tool to remember new things. Without a persona the
//  chat behaves exactly as before.
//

import Foundation

enum ModelPersonaTool {
    static let name = "model_memory_write"

    static var definition: AgentToolDefinition {
        AgentToolDefinition(
            name: name,
            description: "Save a long-term memory of YOUR OWN (this model's persona memory, separate from the shared daily memory log). Use it for things you personally want to remember about the user, your relationship with them, promises you made, or facts about yourself. Keep each entry to one short sentence. Never store passwords, keys or other secrets.",
            parameters: [
                "tool_title": AgentToolParam(type: .string, description: "A concise 5-10 word summary of what is being remembered, shown to the user. Use the same language as the user."),
                "content": AgentToolParam(type: .string, description: "The memory, one short sentence."),
            ],
            required: ["tool_title", "content"],
            propertyOrdering: ["tool_title", "content"]
        )
    }
}

extension AIChatViewModel {

    /// The persona of the model this chat is talking to, if enabled.
    var activeModelPersona: ModelPersona? {
        guard helperConfig == nil else { return nil }
        return ModelPersonaStore.shared.activePersona(for: resolveCurrentEntry()?.id)
    }

    /// Identity section of the system prompt: the model's persona when it has
    /// one, otherwise the global SOUL.md identity.
    func personaAwareIdentitySection() -> String {
        guard let persona = activeModelPersona, let entry = resolveCurrentEntry() else {
            return SystemPromptBuilder.identitySection()
        }
        let name = persona.effectiveName ?? entry.model.displayName
        let identity = "You are \(name), a capable AI assistant running on an iOS device with a fully functional iSH Linux shell (Alpine Linux, aarch64)."
        let section = ModelPersonaStore.promptSection(
            persona, fallbackName: name,
            memoryInstruction: "To save a new long-term memory of your own (about the user, your relationship, or yourself), call the `\(ModelPersonaTool.name)` tool. Use it sparingly, for things worth keeping."
        )
        return section.isEmpty ? identity + " " : identity + "\n\n" + section + "\n\n"
    }

    /// Persona tool definitions to add to this run's tool list.
    func modelPersonaTools() -> [AgentToolDefinition] {
        guard let persona = activeModelPersona, persona.autoMemory else { return [] }
        return [ModelPersonaTool.definition]
    }

    /// Handler for `model_memory_write`.
    func executeModelMemoryWrite(args: [String: Any]) -> (output: String, success: Bool) {
        guard let entry = resolveCurrentEntry(), activeModelPersona != nil else {
            return ("Error: this model has no persona memory enabled.", false)
        }
        let text = (args["content"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return ("Error: 'content' is empty.", false) }
        let saved = ModelPersonaStore.shared.addMemory(entryId: entry.id, text: text, source: .model,
                                                       origin: AppLocalized("Chat"))
        return saved
            ? ("Saved to your memory: \(text)", true)
            : ("Error: automatic memory is turned off for this model.", false)
    }
}
