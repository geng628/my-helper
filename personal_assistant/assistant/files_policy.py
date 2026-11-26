"""文件访问白名单策略与原子写入原语（零内部依赖）。"""
from __future__ import annotations

import os
import tempfile
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REJECT_SEGMENTS = (".env", ".git", "state", ".ssh", "uploads", "models.dsl", "models.json")


# 剥离零宽等 Unicode 格式控制字符后执行 NFKC 规范化与 casefold，防御 APFS 字符折叠绕过
def normalize_segment(part: str) -> str:
    cleaned = "".join(ch for ch in part if unicodedata.category(ch) != "Cf")
    return unicodedata.normalize("NFKC", cleaned).casefold()


# 判断单个路径段是否命中受保护敏感名称（Unicode 折叠等价类拦截，仅精确放行 .env.example）
def is_sensitive_segment(
    part: str,
    reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS,
) -> bool:
    if part == ".env.example":
        return False
    folded = normalize_segment(part)
    if not folded:
        return True
    rejected = {normalize_segment(seg) for seg in reject_segments}
    return (
        folded in rejected
        or folded.startswith(".env.")
        or folded.startswith(".tmp-models-")
    )


# 安全记录路径自身（lstat）与可选解引用目标（stat）的内核设备号与 Inode 元组
def _record_path_inodes(
    path: Path,
    inodes: set[tuple[int, int]],
    *,
    follow_symlinks: bool = True,
) -> None:
    try:
        lst = os.lstat(path)
        inodes.add((lst.st_dev, lst.st_ino))
    except OSError:
        pass
    if follow_symlinks:
        try:
            st = os.stat(path)
            inodes.add((st.st_dev, st.st_ino))
        except OSError:
            pass


# 扫描给定基础目录第一层及已知受保护子项，收集所有受保护文件的 (st_dev, st_ino) 集合
def collect_protected_inodes(
    roots: Iterable[Path],
    reject_segments: tuple[str, ...] = DEFAULT_REJECT_SEGMENTS,
) -> set[tuple[int, int]]:
    inodes: set[tuple[int, int]] = set()
    visited_dirs: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if is_sensitive_segment(root.name, reject_segments):
            _record_path_inodes(root, inodes)
        try:
            key = root.resolve(strict=False)
        except OSError:
            key = root
        if key in visited_dirs:
            continue
        visited_dirs.add(key)

        try:
            with os.scandir(root) as it:
                for entry in it:
                    if not is_sensitive_segment(entry.name, reject_segments):
                        continue
                    entry_path = Path(entry.path)
                    _record_path_inodes(entry_path, inodes)
                    # 对敏感目录（如 state/.ssh/.git/uploads）有界递归收集常规子项 inode，阻断多层硬链接绕过
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            budget = 512
                            stack = [entry_path]
                            while stack and budget > 0:
                                cur_dir = stack.pop()
                                with os.scandir(cur_dir) as sub_it:
                                    for sub in sub_it:
                                        if sub.is_symlink():
                                            continue
                                        sub_path = Path(sub.path)
                                        _record_path_inodes(sub_path, inodes, follow_symlinks=False)
                                        budget -= 1
                                        if budget <= 0:
                                            break
                                        if sub.is_dir(follow_symlinks=False):
                                            stack.append(sub_path)
                    except OSError:
                        pass
        except OSError:
            pass

        for seg in reject_segments:
            _record_path_inodes(root / seg, inodes)
    return inodes


# 沿候选路径及其逐级父目录（直至 stop_roots）比对 stat/lstat 是否落入受保护 Inode 集合
def hits_protected_inode(
    candidate: Path,
    stop_roots: Iterable[Path],
    protected_inodes: set[tuple[int, int]],
) -> bool:
    if not protected_inodes:
        return False

    stop_paths: set[Path] = set()
    stop_inodes: set[tuple[int, int]] = set()
    for raw_stop in stop_roots:
        stop_paths.add(Path(os.path.abspath(raw_stop)))
        try:
            stop_paths.add(Path(raw_stop).resolve(strict=False))
        except OSError:
            pass
        try:
            st = os.stat(raw_stop)
            inode = (st.st_dev, st.st_ino)
            if inode not in protected_inodes:
                stop_inodes.add(inode)
        except OSError:
            pass

    start_chains: list[Path] = [Path(os.path.abspath(candidate))]
    try:
        resolved_cand = Path(candidate).resolve(strict=False)
        if resolved_cand != start_chains[0]:
            start_chains.append(resolved_cand)
    except OSError:
        pass

    for start in start_chains:
        cur = start
        while True:
            if cur in stop_paths:
                break
            try:
                lst = os.lstat(cur)
                if (lst.st_dev, lst.st_ino) in protected_inodes:
                    return True
            except OSError:
                pass
            try:
                st = os.stat(cur)
                inode = (st.st_dev, st.st_ino)
                if inode in protected_inodes:
                    return True
                if inode in stop_inodes:
                    break
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


# 将配置里的相对目录安全解析为绝对路径，data/* 前缀自动映射到自定义 data_root
def _resolve_config_dir(entry: object, project_root: Path, data_root: Path | None) -> Path:
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("目录配置必须是非空字符串。")
    rel = Path(entry.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"目录配置必须是相对路径且不能包含 ..：{entry}")
    if data_root is not None and rel.parts and rel.parts[0] == "data":
        resolved_data = data_root.resolve()
        sub = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")
        target = resolved_data / sub
        if not target.resolve().is_relative_to(resolved_data):
            raise ValueError(f"目录配置超出 data_root 范围：{entry}")
        target.mkdir(parents=True, exist_ok=True)
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
        cls,
        raw: dict[str, object] | None,
        project_root: Path,
        data_root: Path | None = None,
    ) -> "FilesPolicy":
        root = project_root.resolve()
        cfg = raw or {}
        if not isinstance(cfg, dict):
            raise ValueError("files 配置必须是 JSON 对象。")
        raw_reads = cfg.get("read_roots", ["data/memory", "data/workspace", "docs"])
        if not isinstance(raw_reads, list) or not raw_reads:
            raise ValueError("files.read_roots 必须是非空数组。")
        read_roots = tuple(_resolve_config_dir(item, root, data_root) for item in raw_reads)
        write_root = _resolve_config_dir(cfg.get("write_root", "data/workspace"), root, data_root)
        raw_rejects = cfg.get("reject_segments", list(DEFAULT_REJECT_SEGMENTS))
        if not isinstance(raw_rejects, list) or not all(isinstance(x, str) and x for x in raw_rejects):
            raise ValueError("files.reject_segments 必须是字符串数组。")
        return cls(
            project_root=root,
            read_roots=read_roots,
            write_root=write_root,
            max_read_bytes=_positive_int(cfg.get("max_read_bytes"), 65536, "max_read_bytes"),
            max_write_bytes=_positive_int(cfg.get("max_write_bytes"), 65536, "max_write_bytes"),
            max_lines=_positive_int(cfg.get("max_lines"), 400, "max_lines"),
            max_results=_positive_int(cfg.get("max_results"), 100, "max_results"),
            reject_segments=tuple(raw_rejects),
        )

    # 预先整理所有允许根目录的（逻辑路径、真实路径、展示前缀、短别名）映射表
    def build_roots(self, data_root: Path) -> list[tuple[Path, Path, str, str]]:
        seen: set[Path] = set()
        entries: list[tuple[Path, Path, str, str]] = []
        for raw in (*self.read_roots, self.write_root):
            resolved = raw.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append((raw, resolved, *self.root_labels(raw, data_root)))
        return entries

    # 计算单个允许根目录的标准展示路径（如 data/memory）和简写别名（如 memory）
    def root_labels(self, raw_root: Path, data_root: Path) -> tuple[str, str]:
        data_resolved = data_root.resolve()
        proj_resolved = self.project_root.resolve()
        for base, prefix in ((data_resolved, "data"), (proj_resolved, "")):
            try:
                rel = raw_root.relative_to(base)
            except ValueError:
                continue
            if prefix:
                display = ("data" / rel).as_posix() if rel.parts else "data"
                return display, rel.as_posix() if rel.parts else ""
            display = rel.as_posix() if rel.parts else "."
            return display, display
        return raw_root.name, raw_root.name

    # 按允许根目录把磁盘绝对路径格式化为统一的相对展示路径
    def to_display_path(
        self,
        resolved: Path,
        data_root: Path,
        *,
        roots: list[tuple[Path, Path, str, str]] | None = None,
    ) -> str:
        active_roots = roots if roots is not None else self.build_roots(data_root)
        for _, root_resolved, display_prefix, _ in active_roots:
            if resolved.is_relative_to(root_resolved):
                rel = resolved.relative_to(root_resolved)
                return f"{display_prefix}/{rel.as_posix()}" if rel.parts else display_prefix
        data_resolved = data_root.resolve()
        if resolved.is_relative_to(data_resolved):
            return ("data" / resolved.relative_to(data_resolved)).as_posix()
        return resolved.name

    # 检查路径中是否包含受保护的敏感目录段（全量 Unicode 折叠等价类防护，如 .ENV、ſtate、modelſ.json）
    def check_segments(self, parts: tuple[str, ...]) -> None:
        for part in parts:
            if is_sensitive_segment(part, self.reject_segments):
                raise ValueError(f"禁止访问受保护路径：{part}")

    # 收集当前策略与数据目录下的受保护内核 Inode 集合及停止边界
    def _protected_context(self, data_root: Path) -> tuple[list[Path], set[tuple[int, int]]]:
        proj_resolved = self.project_root.resolve(strict=False)
        data_resolved = data_root.resolve(strict=False)
        write_resolved = self.write_root.resolve(strict=False)
        scan_roots: list[Path] = [
            proj_resolved,
            proj_resolved / "config",
            proj_resolved / "data",
            data_resolved,
            *(raw.resolve(strict=False) for raw in self.read_roots),
            write_resolved,
            write_resolved.parent,
        ]
        stop_roots: list[Path] = [
            proj_resolved,
            data_resolved,
            *(raw.resolve(strict=False).parent for raw in (*self.read_roots, self.write_root)),
        ]
        return stop_roots, collect_protected_inodes(scan_roots, self.reject_segments)

    # 逐级检验只读候选路径的符号链接与 Inode，拦截中途逃逸允许目录或指向受保护目标
    def _validate_read_candidate(
        self,
        candidate: Path,
        data_root: Path,
        stop_roots: list[Path],
        protected_inodes: set[tuple[int, int]],
    ) -> None:
        if hits_protected_inode(candidate, stop_roots, protected_inodes):
            raise ValueError("禁止访问受保护路径。")

        resolved_read_roots = tuple(r.resolve() for r in self.read_roots)
        base_candidates = (*self.read_roots, data_root.resolve(), self.project_root.resolve())
        base_root: Path | None = None
        rel_parts: tuple[str, ...] = ()
        for base in base_candidates:
            try:
                rel_parts = candidate.relative_to(base).parts
                base_root = base
                break
            except ValueError:
                continue

        if base_root is None:
            return
        current = base_root
        for part in rel_parts:
            current = current / part
            if is_sensitive_segment(part, self.reject_segments):
                raise ValueError(f"禁止访问受保护路径：{part}")
            if current.is_symlink():
                target = current.resolve(strict=False)
                if hits_protected_inode(target, stop_roots, protected_inodes):
                    raise ValueError(f"读取路径符号链接指向受保护目标：{part}")
                matched_root = next((rr for rr in resolved_read_roots if target.is_relative_to(rr)), None)
                if matched_root is None:
                    raise ValueError(f"读取路径符号链接超出允许目录范围：{part}")
                self.check_segments(target.relative_to(matched_root).parts)

    # 安全解析相对路径：校验读写白名单、拒绝 .. 穿越、敏感目录（含 Unicode/Inode）与越权软链接
    def resolve_path(
        self,
        rel_path: str,
        data_root: Path,
        *,
        write: bool = False,
        roots: list[tuple[Path, Path, str, str]] | None = None,
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

        active_roots = roots if roots is not None else self.build_roots(data_root)
        candidate = self._map_read_candidate(rel, data_root, active_roots)
        self._validate_read_candidate(candidate, data_root, stop_roots, protected_inodes)

        resolved = candidate.resolve(strict=False)
        if hits_protected_inode(resolved, stop_roots, protected_inodes):
            raise ValueError("禁止访问受保护路径。")
        for raw_root in self.read_roots:
            root_resolved = raw_root.resolve()
            if resolved.is_relative_to(root_resolved):
                self.check_segments(resolved.relative_to(root_resolved).parts)
                return resolved
        allowed = "、".join(self.root_labels(r, data_root)[0] for r in self.read_roots)
        raise ValueError(f"路径不在允许读取的目录内；允许范围：{allowed}")

    # 解析并校验写入路径：限制在 write_root（data/workspace）内且禁止任何软链接组件及受保护 Inode
    def _resolve_write_path(
        self,
        rel: Path,
        data_root: Path,
        stop_roots: list[Path],
        protected_inodes: set[tuple[int, int]],
    ) -> Path:
        parts = rel.parts
        write_root = self.write_root
        write_resolved = write_root.resolve()
        write_display, write_alias = self.root_labels(write_root, data_root)
        display_parts = Path(write_display).parts

        if parts[: len(display_parts)] == display_parts:
            unresolved = write_root / Path(*parts[len(display_parts) :])
        elif write_alias and parts[0] == write_alias:
            unresolved = write_root / Path(*parts[1:])
        elif self._matches_other_root(parts, data_root):
            raise ValueError(f"只能写入 {write_display} 目录。")
        else:
            unresolved = write_root / rel

        if hits_protected_inode(unresolved, stop_roots, protected_inodes):
            raise ValueError("禁止写入受保护路径。")

        current = write_root
        for part in unresolved.relative_to(write_root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("写入路径不能包含符号链接。")
        resolved = unresolved.resolve(strict=False)
        if hits_protected_inode(resolved, stop_roots, protected_inodes):
            raise ValueError("禁止写入受保护路径。")
        if not resolved.is_relative_to(write_resolved):
            raise ValueError(f"只能写入 {write_display} 目录。")
        self.check_segments(resolved.relative_to(write_resolved).parts)
        return resolved

    # 判断写入路径是否误指向其他只读根目录（如 data/memory、docs 等）
    def _matches_other_root(self, parts: tuple[str, ...], data_root: Path) -> bool:
        if not parts:
            return False
        if parts[0] == "data":
            return True
        for raw_root in self.read_roots:
            display, alias = self.root_labels(raw_root, data_root)
            d_parts = Path(display).parts
            if parts[: len(d_parts)] == d_parts or (alias and parts[0] == Path(alias).parts[0]):
                return True
        return False

    # 将用户传入的读取路径（支持 data/memory/... 或简写 memory/...）映射到磁盘路径
    def _map_read_candidate(
        self,
        rel: Path,
        data_root: Path,
        roots: list[tuple[Path, Path, str, str]],
    ) -> Path:
        parts = rel.parts
        for raw_root, _, display_prefix, short_alias in roots:
            d_parts = Path(display_prefix).parts
            if parts[: len(d_parts)] == d_parts:
                return raw_root / Path(*parts[len(d_parts) :])
            if short_alias:
                a_parts = Path(short_alias).parts
                if parts[: len(a_parts)] == a_parts:
                    return raw_root / Path(*parts[len(a_parts) :])
        if parts and parts[0] == "data":
            return data_root.resolve() / Path(*parts[1:])
        return self.project_root.resolve() / rel


# 通过同目录临时文件 + os.replace 实现原子写入，防止写到一半断电损坏文件
def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
