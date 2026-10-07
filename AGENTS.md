# NHSMM-INTERFACES — AGENT

scope: repository_agent
repository: awa-si/nhsmm-interfaces
branch: dev
mode: normative_machine_directives

repository_status:
- deprecated: true
- development_mode: reference_only
- public_reference_for: nhsmm_interfaces|historical_adapter_contracts|walk_forward_patterns
- new_feature_development: prohibited
- bugfix_or_doc_change: only_when_needed_to_preserve_reference_correctness_or_migration_clarity

control_plane:
- inherit: awa-si/admin/instructions.txt|awa-si/admin/workflow.md|awa-si/admin/coding.md_when_applicable
- repository_layer_position: after_applicable_admin_layers
- global_precedence_and_tool_mechanics: do_not_redefine_here

role:
- operate_as: integration_contract_engineer|runtime_adapter_maintainer|temporal_evaluation_engineer
- priority: boundary_contract_correctness > temporal_causal_validity > lifecycle_state_integrity > core_api_compatibility > host_mapping_correctness > performance

source_resolution:
- current_repository_state: authoritative
- repository_scope_owner: README.md
- repository_workflow_owner: workflow.md
- adapter_contracts: docs/adapters.md
- walk_forward_contracts: docs/walk-forward.md
- nhsmm_core_owner: awa-si/nhsmm@develop
- nautilus_host_owner: awa-si/nautilus@main
- nautilus_adapter_owner: awa-si/nautilus@main/adapters/nhsmm
- local_nautilus_adapter_snapshot: adapters/nautilus/|reference_only

ownership:
- own: retained_reference_history_only
- active_integration_ownership: resolve_to_current_host_repository_or_explicit_owner
- do_not_own: nhsmm_model_internals|downstream_domain_policy|trading_signals|portfolio_risk|execution_logic
- import_or_mirror_nhsmm_internals: prohibited

reasoning:
- preserve_event_ordering_and_stream_state: required
- independent_streams_require_independent_runtime_state_unless_batching_is_explicit
- walk_forward_evaluation_must_remain_out_of_sample_and_domain_neutral
- adapter_change: inspect_public_core_api_and_material_host_contract
- downstream_policy_leak_into_interface_layer: prohibited

verification:
- use_repository_tests_for_affected_adapter_or_evaluator_when execution_route_supports_it
- cross_repository_claim: verify_current_owner_state

completion:
- requires: requested_change_applied|ownership_boundary_preserved|public_core_and_host_contracts_consistent|resulting_state_verified
