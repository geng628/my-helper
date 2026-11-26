"""文件访问白名单策略与受保护 Inode 校验底座（零内部依赖）。"""
from __future__ import annotations

import os
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REJECT_SEGMENTS = (".env", ".git", "state", ".ssh", "uploads", "models.dsl", "models.json")


# 剥离零宽等格式控制字符后执行 NFKC 规范化与 casefold，防御 APFS 字符折叠绕过
def normalize_segment(part: str) -> str:
    cleaned = "".join(ch for ch in part if unicodedata.category(ch) != "Cf")
    return unicodedata.normalize("NFKC", cleaned).casefold()


# 判断单个路径段是否命中受保护敏感名称（仅精确放行 .env.example，并屏蔽 .env.* 与原子写盘 .tmp-* 暂存文件）
def is_sensitive_segment(part: str, reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS) -> bool:
    if part == ".env.example":
        return False
    folded = normalize_segment(part)
    if not folded:
        return True
    rejected = {normalize_segment(seg) for seg in reject_segments}
    return folded in rejected or folded.startswith((".env.", ".tmp-"))


# 安全记录路径自身（lstat）与解引用目标（stat）的设备号与 Inode 元组
def _record_path_inodes(path: Path, inodes: set[tuple[int, int]]) -> None:
    for stat_fn in (os.lstat, os.stat):
        try:
            st = stat_fn(path)
            inodes.add((st.st_dev, st.st_ino))
        except OSError:
            pass


# 扫描基础目录单层子项及敏感目录直接子文件，收集受保护 (st_dev, st_ino) 集合
def collect_protected_inodes(
    roots: Iterable[Path], reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS
) -> set[tuple[int, int]]:
    inodes: set[tuple[int, int]] = set()
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if is_sensitive_segment(root.name, reject_segments):
            _record_path_inodes(root, inodes)
        key = root.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        for seg in reject_segments:
            _record_path_inodes(root / seg, inodes)
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if not is_sensitive_segment(entry.name, reject_segments):
                        continue
                    _record_path_inodes(Path(entry.path), inodes)
                    if entry.is_dir(follow_symlinks=False):
                        try:
                            with os.scandir(entry.path) as sub_it:
                                for sub in sub_it:
                                    _record_path_inodes(Path(sub.path), inodes)
                        except OSError:
                            pass
        except OSError:
            pass
    return inodes


# 沿候选路径向父目录回溯至 stop_roots，检查 lstat/stat 是否命中受保护 Inode
def hits_protected_inode(
    candidate: Path, stop_roots: Iterable[Path], protected_inodes: set[tuple[int, int]]
) -> bool:
    if not protected_inodes:
        return False
    stops = {p for raw in stop_roots for p in (Path(os.path.abspath(raw)), Path(raw).resolve(strict=False))}
    cur = Path(os.path.abspath(candidate))
    while cur not in stops:
        for stat_fn in (os.lstat, os.stat):
            try:
                st = stat_fn(cur)
                if (st.st_dev, st.st_ino) in protected_inodes:
                    return True
            except OSError:
                pass
        if cur.parent == cur:
            break
        cur = cur.parent
    return False


# 校验配置中的整数项必须为正整数，缺省时返回默认值
def _positive_int(value: object, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return value


# 将配置相对目录安全解析为绝对路径（纯计算无 mkdir 副作用，data/* 自动映射到自定义 data_root）
def _resolve_config_dir(entry: object, project_root: Path, data_root: Path | None) -> Path:
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("目录配置必须是非空字符串。")
    rel = Path(entry.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"目录配置必须是相对路径且不能包含 ..：{entry}")
    if data_root is not None and rel.parts and rel.parts[0] == "data":
        resolved_data = data_root.resolve()
        target = resolved_data / (Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path("."))
        if not target.resolve().is_relative_to(resolved_data):
            raise ValueError(f"目录配置超出 data_root 范围：{entry}")
        return target
    target = project_root / rel
    if not target.resolve().is_relative_to(project_root.resolve()):
        raise ValueError(f"目录配置超出项目范围：{entry}")
    return target


# 文件访问白名单策略（允许读写根目录、行数/字节限制、路径解析校验与禁止访问目录段）
@dataclass(frozen=True)
class FilesPolicy:
    project_root: Path
    read_roots: tuple[Path, ...]
    write_root: Path
    max_read_bytes: int = 65536
    max_write_bytes: int = 65536
    max_lines: int = 400
    max_results: int = 100
    reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS

    # 从 tools.json 的 files 字段字典构造并校验 FilesPolicy
    @classmethod
    def from_dict(
        cls, raw: dict[str, object] | None, project_root: Path, data_root: Path | None = None
    ) -> "FilesPolicy":
        root = project_root.resolve()
        cfg = raw or {}
        if not isinstance(cfg, dict):
            raise ValueError("files 配置必须是 JSON 对象。")
        raw_reads = cfg.get("read_roots", ["data/memory", "data/workspace", "docs"])
        if not isinstance(raw_reads, list) or not raw_reads:
            raise ValueError("files.read_roots 必须是非空数组。")
        raw_rejects = cfg.get("reject_segments", list(DEFAULT_REJECT_SEGMENTS))
        if not isinstance(raw_rejects, list) or not all(isinstance(x, str) and x for x in raw_rejects):
            raise ValueError("files.reject_segments 必须是字符串数组。")
        return cls(
            project_root=root,
            read_roots=tuple(_resolve_config_dir(item, root, data_root) for item in raw_reads),
            write_root=_resolve_config_dir(cfg.get("write_root", "data/workspace"), root, data_root),
            max_read_bytes=_positive_int(cfg.get("max_read_bytes"), 65536, "max_read_bytes"),
            max_write_bytes=_positive_int(cfg.get("max_write_bytes"), 65536, "max_write_bytes"),
            max_lines=_positive_int(cfg.get("max_lines"), 400, "max_lines"),
            max_results=_positive_int(cfg.get("max_results"), 100, "max_results"),
            reject_segments=tuple(raw_rejects),
        )

    # 整理允许根目录的（逻辑路径、真实路径、规范展示前缀）映射表
    def build_roots(self, data_root: Path) -> list[tuple[Path, Path, str]]:
        seen: set[Path] = set()
        entries: list[tuple[Path, Path, str]] = []
        for raw in (*self.read_roots, self.write_root):
            resolved = raw.resolve()
            if resolved not in seen:
                seen.add(resolved)
                entries.append((raw, resolved, self.root_label(raw, data_root)))
        return entries

    # 计算单个允许根目录的规范相对展示路径（如 data/memory、data/workspace、docs）
    def root_label(self, raw_root: Path, data_root: Path) -> str:
        for base, prefix in ((data_root.resolve(), "data"), (self.project_root.resolve(), "")):
            if raw_root.is_relative_to(base):
                rel = raw_root.relative_to(base)
                if prefix:
                    return ("data" / rel).as_posix() if rel.parts else "data"
                return rel.as_posix() if rel.parts else "."
        return raw_root.name

    # 按允许根目录把磁盘绝对路径格式化为统一的规范相对展示路径
    def to_display_path(
        self, resolved: Path, data_root: Path, *, roots: list[tuple[Path, Path, str]] | None = None
    ) -> str:
        for _, root_resolved, display_prefix in roots or self.build_roots(data_root):
            if resolved.is_relative_to(root_resolved):
                rel = resolved.relative_to(root_resolved)
                return f"{display_prefix}/{rel.as_posix()}" if rel.parts else display_prefix
        data_resolved = data_root.resolve()
        if resolved.is_relative_to(data_resolved):
            return ("data" / resolved.relative_to(data_resolved)).as_posix()
        return resolved.name

    # 检查路径段是否包含受保护敏感名称（含 Unicode 折叠等价类拦截）
    def check_segments(self, parts: tuple[str, ...]) -> None:
        for part in parts:
            if is_sensitive_segment(part, self.reject_segments):
                raise ValueError(f"禁止访问受保护路径：{part}")

    # 收集当前策略与数据目录下的停止边界及受保护 Inode 集合
    def _protected_context(self, data_root: Path) -> tuple[list[Path], set[tuple[int, int]]]:
        proj, data = self.project_root.resolve(strict=False), data_root.resolve(strict=False)
        all_roots = [r.resolve(strict=False) for r in (*self.read_roots, self.write_root)]
        scan_roots = [proj, proj / "config", proj / "data", data, *all_roots, *(r.parent for r in all_roots)]
        stop_roots = [proj, data, *(r.parent for r in all_roots)]
        return stop_roots, collect_protected_inodes(scan_roots, self.reject_segments)

    # 安全解析相对路径：校验读写白名单、拒绝 .. 穿越、敏感段与越权软链接
    def resolve_path(
        self,
        rel_path: str,
        data_root: Path,
        *,
        write: bool = False,
        roots: list[tuple[Path, Path, str]] | None = None,
    ) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip() or "\x00" in rel_path:
            raise ValueError("路径必须是非空字符串。")
        rel = Path(rel_path.strip().replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("路径必须是项目内相对路径，且不能包含 ..。")
        self.check_segments(rel.parts)
        stop_roots, protected_inodes = self._protected_context(data_root)
        if write:
            return self._resolve_write_path(rel, data_root, stop_roots, protected_inodes)
        return self._resolve_read_path(rel, data_root, roots or self.build_roots(data_root), stop_roots, protected_inodes)

    # 解析并逐级校验只读路径：拦截中途软链逃逸再折返或指向受保护 Inode
    def _resolve_read_path(
        self,
        rel: Path,
        data_root: Path,
        roots: list[tuple[Path, Path, str]],
        stop_roots: list[Path],
        protected_inodes: set[tuple[int, int]],
    ) -> Path:
        candidate = self._map_read_candidate(rel, data_root, roots)
        if hits_protected_inode(candidate, stop_roots, protected_inodes):
            raise ValueError("禁止访问受保护路径。")
        read_roots = tuple(r.resolve() for r in self.read_roots)
        bases = (*self.read_roots, data_root.resolve(), self.project_root.resolve())
        base = next((b for b in bases if candidate.is_relative_to(b)), None)
        if base is not None:
            cur = base
            for part in candidate.relative_to(base).parts:
                self.check_segments((part,))
                cur = cur / part
                if cur.is_symlink():
                    target = cur.resolve(strict=False)
                    if hits_protected_inode(target, stop_roots, protected_inodes):
                        raise ValueError(f"读取路径符号链接指向受保护目标：{part}")
                    owner = next((rr for rr in read_roots if target.is_relative_to(rr)), None)
                    if owner is None:
                        raise ValueError(f"读取路径符号链接超出允许目录范围：{part}")
                    self.check_segments(target.relative_to(owner).parts)

        resolved = candidate.resolve(strict=False)
        if hits_protected_inode(resolved, stop_roots, protected_inodes):
            raise ValueError("禁止访问受保护路径。")
        owner = next((rr for rr in read_roots if resolved.is_relative_to(rr)), None)
        if owner is None:
            allowed = "、".join(self.root_label(r, data_root) for r in self.read_roots)
            raise ValueError(f"路径不在允许读取的目录内；允许范围：{allowed}")
        self.check_segments(resolved.relative_to(owner).parts)
        return resolved

    # 解析并校验写入路径：支持 data/workspace/<subpath> 或工作区内相对路径，拦截 data/<非workspace> 与符号链接
    def _resolve_write_path(
        self, rel: Path, data_root: Path, stop_roots: list[Path], protected_inodes: set[tuple[int, int]]
    ) -> Path:
        parts = rel.parts
        write_resolved = self.write_root.resolve()
        write_display = self.root_label(self.write_root, data_root)
        d_parts = Path(write_display).parts
        if d_parts and parts[: len(d_parts)] == d_parts:
            sub = Path(*parts[len(d_parts) :])
        elif parts and parts[0] == "data":
            raise ValueError(f"只能写入 {write_display} 目录。")
        else:
            sub = rel
        if not sub.parts:
            raise ValueError("写入路径必须指定具体文件。")

        unresolved = self.write_root / sub
        cur = self.write_root
        for part in sub.parts:
            cur = cur / part
            if cur.is_symlink():
                raise ValueError("写入路径不能包含符号链接。")
        resolved = unresolved.resolve(strict=False)
        if hits_protected_inode(unresolved, stop_roots, protected_inodes) or hits_protected_inode(
            resolved, stop_roots, protected_inodes
        ):
            raise ValueError("禁止写入受保护路径。")
        if not resolved.is_relative_to(write_resolved):
            raise ValueError(f"只能写入 {write_display} 目录。")
        self.check_segments(resolved.relative_to(write_resolved).parts)
        return resolved

    # 将用户传入的规范相对读取路径映射到候选磁盘路径
    def _map_read_candidate(
        self, rel: Path, data_root: Path, roots: list[tuple[Path, Path, str]]
    ) -> Path:
        parts = rel.parts
        for raw_root, _, display_prefix in roots:
            p_parts = Path(display_prefix).parts
            if p_parts and parts[: len(p_parts)] == p_parts:
                return raw_root / Path(*parts[len(p_parts) :])
        if parts and parts[0] == "data":
            return data_root.resolve() / Path(*parts[1:])
        return self.project_root.resolve() / rel
