# Vocatio Project Intent

## Current Decision

`vocatio` will become a new, dedicated system for managing event media workflows.

The current script-based workflow in `scriptoza` is already more complex than the rest of the utility scripts in that repository, so it should not continue to grow there indefinitely.

For now:

- `scriptoza` remains the active production tool
- current deliveries continue to be handled manually, as they are today
- `vocatio` is reserved for the future integrated system
- no implementation work is required here yet beyond documenting the intended direction

## Product Goal

The goal is to build one coherent system that covers:

- ingesting and processing event media
- identifying performances from video announcements and timestamps
- assigning photos and video clips to performance sets
- reviewing and correcting the automatic results
- preparing publishable delivery assets
- sharing those assets with recipients through a web frontend

This is not meant to be a loose set of scripts.
It is meant to become an operator-controlled media workflow system.

## Main Roles

### 1. Operator GUI

The desktop GUI is for internal use only.

It is the main control center for the whole workflow.

The operator GUI should allow step-by-step control over:

- scanning media for a day
- syncing video streams
- running transcription
- extracting announcement candidates
- building performance timelines
- assigning photos
- generating proxy assets
- reviewing and correcting results
- preparing final deliveries
- publishing recipient-facing galleries

The GUI is not just a viewer.
It is the workflow controller and the place where business decisions are made.

### 2. Recipient Web Frontend

The web frontend is for recipients only.

It should not contain business logic or manual review logic.
It should only expose what the operator already approved and published from the GUI.

Examples:

- private gallery view
- access by individual token
- image preview
- download access
- optionally video preview or short delivery clips

The web frontend is a presentation and access layer, not an editorial tool.

## Core Product Principle

The GUI is the source of operational intent.

The web frontend is the delivery surface.

That means:

- the operator decides what belongs to a set
- the operator decides what should be published
- the operator decides which assets are visible
- the operator generates or approves access tokens
- the web app simply reflects those published decisions

## Intended Publish Workflow

The future system should support a workflow like this:

1. The operator reviews a performance set in the GUI.
2. The operator decides that the set is ready for delivery.
3. The operator clicks `Publish`.
4. The system runs the necessary background jobs.

Expected effects of `Publish`:

- generate delivery JPG files from originals
- apply export presets such as target size and JPEG quality
- optionally generate derived video files
- place generated delivery files in the correct publish location
- create a delivery record
- generate an access token
- build a ready-to-copy message from a template
- expose the published gallery in the web frontend

Example outcome visible in the GUI:

- delivery URL
- access token or access code
- generated message text ready to copy into Instagram, Messenger, email, etc.

## Message Templates

The system should later support message templates for recipients.

Example concept:

- recipient name
- delivery link
- access code
- optional delivery note

The operator should be able to copy a generated message after publishing.

## Functional Scope

The future system is expected to cover at least:

- photo workflows
- video workflows
- performance assignment
- review and correction
- export generation
- access control
- web sharing

This means the final system is broader than the current photo assignment and review tooling.

## Migration Strategy

Migration should happen gradually and intentionally.

The current plan is:

1. Continue finishing and stabilizing the current script-based workflow in `scriptoza`.
2. Use that work as the reference behavior.
3. Design the target architecture in `vocatio`.
4. Rebuild the system cleanly instead of indefinitely extending the old script repository.

Important constraint:

- do not rush implementation in `vocatio`
- first finish the current working script flow
- then use those proven steps as the basis for the new system

## Architectural Direction

The future system will likely contain:

- a Python backend / service layer
- a desktop GUI for the operator
- a web frontend for recipients
- background jobs for heavy media processing
- persistent structured state instead of ad hoc JSON-only workflow state

The current script logic should later be migrated into reusable services or pipeline modules rather than copied as standalone scripts.

## What Is Not Being Done Yet

At this stage, `vocatio` is not yet being implemented as a full application.

For now, the repository exists only to capture direction and to prepare for a clean rebuild later.

Current non-goals:

- no early framework lock-in yet
- no premature database schema yet
- no partial rewrite of the current workflow yet
- no migration before the existing script project is sufficiently complete

## Summary

`vocatio` is planned as a dedicated event media workflow system with:

- desktop operator GUI for full control
- automated media processing pipelines
- recipient-facing web delivery
- token-based private access
- publish actions driven from the GUI

The current script-based workflow remains the production path for now.
`vocatio` will be built afterward as a clean, integrated system based on the lessons and logic from that workflow.
